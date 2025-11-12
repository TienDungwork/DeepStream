import sys
import gi
import numpy as np
import cv2

gi.require_version('Gst', '1.0')
from gi.repository import Gst

try:
    import pyds
except ImportError:
    print("WARNING: pyds module not found")
    pyds = None

from typing import Dict, Optional
from multi_camera_app.models import StreamData
from multi_camera_app.utils import PolygonUtils
from multi_camera_app.threads import FrameSaverThread, RTMPStreamManager


class ProbeHandler:
    def __init__(self, streams: Dict[int, StreamData], polygon_utils: PolygonUtils, 
                 frame_saver: FrameSaverThread, save_frames: bool, draw_bbox: bool = True,
                 streammux_width: int = 1920, streammux_height: int = 1080,
                 rtmp_stream_manager: Optional[RTMPStreamManager] = None):
        self.streams = streams
        self.polygon_utils = polygon_utils
        self.frame_saver = frame_saver
        self.save_frames = save_frames
        self.draw_bbox = draw_bbox
        self.streammux_width = streammux_width
        self.streammux_height = streammux_height
        self.rtmp_stream_manager = rtmp_stream_manager
        
        # Precompute tile positions cache for performance
        self.tile_cache = {}
        
        # Cache scaled polygons per stream
        self.scaled_polygons = {}
        
        # Cache stream objects
        self.stream_objects = {}
    
    def _get_scaled_polygon(self, stream_id: int, frame_width: int, frame_height: int) -> list:
        cache_key = (stream_id, frame_width, frame_height)
        
        if cache_key not in self.scaled_polygons:
            if stream_id in self.streams:
                data = self.streams[stream_id]
                scaled = data.scale_polygon_to_resolution(frame_width, frame_height)
                
                # Debug
                if not self.scaled_polygons:
                    print(f"  Original: {data.polygon_original_width}x{data.polygon_original_height}")
                    #print(f"  Target: {frame_width}x{frame_height}")
                    #print(f"  Scale: {frame_width/data.polygon_original_width:.3f}x, {frame_height/data.polygon_original_height:.3f}y")
                    if data.polygon_points:
                        print(f"  First point: {data.polygon_points[0]} -> {scaled[0]}")
                
                self.scaled_polygons[cache_key] = scaled
        
        return self.scaled_polygons.get(cache_key, [])
    
    def pgie_src_pad_buffer_probe(self, pad, info, u_data):
        if not pyds:
            return Gst.PadProbeReturn.OK
            
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            
            stream_id = frame_meta.source_id
            
            if stream_id not in self.streams:
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue
            
            data = self.streams[stream_id]
            data.increment_frame()
            data.update_fps()
            

            frame_width = self.streammux_width
            frame_height = self.streammux_height
            
            scaled_polygon = self._get_scaled_polygon(stream_id, frame_width, frame_height)
            
            l_obj = frame_meta.obj_meta_list
            objects_to_remove = []
            objects_inside = []
            num_total = 0
            
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                
                num_total += 1
                
                center_x = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
                center_y = obj_meta.rect_params.top + obj_meta.rect_params.height / 2
                
                if not self.polygon_utils.is_point_in_polygon(center_x, center_y, scaled_polygon):
                    objects_to_remove.append(obj_meta)
                else:
                    # Only draw bbox if enabled
                    if self.draw_bbox:
                        obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
                        obj_meta.rect_params.border_width = 3
                    
                    class_name = obj_meta.obj_label if obj_meta.obj_label else f"class_{obj_meta.class_id}"
                    
                    objects_inside.append({
                        'id': obj_meta.object_id,
                        'class_id': obj_meta.class_id,
                        'class_name': class_name,
                        'confidence': obj_meta.confidence,
                        'bbox': (int(obj_meta.rect_params.left), 
                                int(obj_meta.rect_params.top),
                                int(obj_meta.rect_params.width), 
                                int(obj_meta.rect_params.height)),
                        'center': (int(center_x), int(center_y))
                    })
                
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            if len(objects_inside) > 0:
                print(f"[{data.name:12s}] Frame {data.frame_count:5d} | FPS: {data.fps:6.2f} | Objects: {len(objects_inside)}/{num_total}")
                
                # Store objects for this stream to use in tiler probe
                self.stream_objects[stream_id] = {
                    'objects': objects_inside,
                    'frame_number': data.frame_count,
                    'stream_name': data.name,
                    'polygon': scaled_polygon
                }
            
            elif data.frame_count % 100 == 0:
                print(f"[{data.name:12s}] Frame {data.frame_count:5d} | FPS: {data.fps:6.2f} | No objects ({num_total} detected)")
            
            for obj_meta in objects_to_remove:
                try:
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                except:
                    pass
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        return Gst.PadProbeReturn.OK
    
    def _get_tile_position(self, stream_id, num_streams, tiler_width, tiler_height, streammux_width, streammux_height):
        cache_key = (stream_id, num_streams, tiler_width, tiler_height, streammux_width, streammux_height)
        if cache_key not in self.tile_cache:
            tiler_rows = int(np.ceil(np.sqrt(num_streams)))
            tiler_cols = int(np.ceil(num_streams / tiler_rows))
            
            tile_width = tiler_width // tiler_cols
            tile_height = tiler_height // tiler_rows
            
            tile_row = stream_id // tiler_cols
            tile_col = stream_id % tiler_cols
            
            offset_x = tile_col * tile_width
            offset_y = tile_row * tile_height
            
            # Scale based on streammux resolution (not hardcoded 1920x1080)
            scale_x = tile_width / streammux_width
            scale_y = tile_height / streammux_height
            
            self.tile_cache[cache_key] = (offset_x, offset_y, scale_x, scale_y, tile_width, tile_height)
        
        return self.tile_cache[cache_key]
    
    def tiler_src_pad_buffer_probe(self, pad, info, u_data, tiler_width, tiler_height, streammux_width, streammux_height):
        if not pyds:
            return Gst.PadProbeReturn.OK
            
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        
        num_streams = len(self.streams)
        
        l_frame = batch_meta.frame_meta_list
        if l_frame is None:
            return Gst.PadProbeReturn.OK
        
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            return Gst.PadProbeReturn.OK
        
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        
        for stream_id in range(num_streams):
            if stream_id not in self.streams:
                continue
                
            data = self.streams[stream_id]
            
            # Use cached tile position with streammux resolution
            offset_x, offset_y, scale_x, scale_y, tile_width, tile_height = self._get_tile_position(
                stream_id, num_streams, tiler_width, tiler_height, streammux_width, streammux_height)
            
            # Get polygon already scaled to streammux resolution
            scaled_polygon = self._get_scaled_polygon(stream_id, streammux_width, streammux_height)
            
            if data.polygon_ready and len(scaled_polygon) >= 3:
                for i in range(len(scaled_polygon)):
                    if display_meta.num_lines >= 16:
                        break
                        
                    p1 = scaled_polygon[i]
                    p2 = scaled_polygon[(i + 1) % len(scaled_polygon)]
                    
                    # Bây giờ chỉ cần scale để fit vào tile và thêm offset
                    line_params = display_meta.line_params[display_meta.num_lines]
                    line_params.x1 = int(p1[0] * scale_x + offset_x)
                    line_params.y1 = int(p1[1] * scale_y + offset_y)
                    line_params.x2 = int(p2[0] * scale_x + offset_x)
                    line_params.y2 = int(p2[1] * scale_y + offset_y)
                    line_params.line_width = 3
                    line_params.line_color.set(0.0, 1.0, 0.0, 1.0)
                    display_meta.num_lines += 1
            
            if display_meta.num_labels < 16:
                text_params = display_meta.text_params[display_meta.num_labels]
                text_params.display_text = f"{data.name} | FPS: {data.fps:.2f}"
                text_params.x_offset = offset_x + 10
                text_params.y_offset = offset_y + 30
                text_params.font_params.font_name = "Serif"
                text_params.font_params.font_size = 14
                text_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)
                text_params.set_bg_clr = 1
                text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
                display_meta.num_labels += 1
            
            if self.frame_saver and self.save_frames and stream_id in self.stream_objects:
                try:
                    stream_data = self.stream_objects[stream_id]
                    
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    frame_rgba = np.array(n_frame, copy=True, order='C')
                    frame_rgba = frame_rgba.reshape((tiler_height, tiler_width, 4))
                    
                    tile_frame = frame_rgba[offset_y:offset_y+tile_height, offset_x:offset_x+tile_width]
                    frame_bgr = cv2.cvtColor(tile_frame, cv2.COLOR_RGBA2BGR)
                    frame_bgr = cv2.resize(frame_bgr, (streammux_width, streammux_height))
                    
                    for obj in stream_data['objects']:
                        x, y, w, h = obj['bbox']
                        x = max(0, x)
                        y = max(0, y)
                        x2 = min(streammux_width, x + w)
                        y2 = min(streammux_height, y + h)
                        
                        if x2 > x and y2 > y:
                            obj_crop = frame_bgr[y:y2, x:x2]
                            
                            self.frame_saver.add_frame(
                                stream_id,
                                stream_data['stream_name'],
                                obj_crop,
                                [obj],
                                stream_data['frame_number'],
                                stream_data['polygon']
                            )
                    
                    del self.stream_objects[stream_id]
                    
                except Exception as e:
                    if data.frame_count % 100 == 1:
                        print(f"Error extracting objects: {type(e).__name__}: {e}")
            
            if self.rtmp_stream_manager:
                try:
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    frame_rgba = np.array(n_frame, copy=True, order='C')
                    frame_rgba = frame_rgba.reshape((tiler_height, tiler_width, 4))
                    
                    tile_frame = frame_rgba[offset_y:offset_y+tile_height, offset_x:offset_x+tile_width]
                    frame_bgr = cv2.cvtColor(tile_frame, cv2.COLOR_RGBA2BGR)
                    
                    metadata = {
                        'stream_id': stream_id,
                        'camera_name': data.name,
                        'fps': data.fps,
                        'polygon': scaled_polygon if data.polygon_ready else [],
                        'objects': self.stream_objects.get(stream_id, {}).get('objects', []),
                        'streammux_width': streammux_width,
                        'streammux_height': streammux_height
                    }
                    
                    self.rtmp_stream_manager.push_frame(stream_id, frame_bgr, metadata)
                    
                except Exception as e:
                    if data.frame_count % 100 == 1:
                        print(f"Error pushing frame to RTMP: {type(e).__name__}: {e}")
        
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
        return Gst.PadProbeReturn.OK
