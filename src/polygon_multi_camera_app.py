import sys
import gi
import cv2
import numpy as np
import json
import os
import threading
import queue
import time

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstVideo, GstRtspServer
import pyds
from shapely.geometry import Point, Polygon


class MultiCameraPolygonApp:
    def __init__(self, streams_config, config_file, rtsp_port=8554, 
                 udp_port_start=5400, save_frames=False, output_dir="output_frames", 
                 draw_bbox=True, skip_prompt=False):
        self.streams_config_file = streams_config
        self.config_file = config_file
        self.rtsp_port = rtsp_port
        self.udp_port_start = udp_port_start
        
        # Frame saving options
        self.save_frames = save_frames
        self.output_dir = output_dir
        self.draw_bbox = draw_bbox
        self.skip_prompt = skip_prompt
        
        # Load streams configuration
        self.streams = []
        self.load_streams_config()
        
        # Per-stream data
        self.stream_data = {}
        for i, stream in enumerate(self.streams):
            self.stream_data[i] = {
                'polygon_points': [],
                'polygon_ready': False,
                'frame_count': 0,
                'fps': 0.0,
                'fps_start_time': None,
                'fps_frame_count': 0,
                'saved_frame_count': 0,
                'name': stream.get('name', f'Camera {i+1}'),
                'polygon_file': stream.get('polygon_file', f'polygon_cam{i+1}.json')
            }
            # Load polygon for this stream
            self.load_polygon(i)
        
        # Threading for async frame saving
        self.frame_queue = queue.Queue(maxsize=60)
        self.save_thread = None
        self.stop_save_thread = False
        
        # Create output directory if needed
        if self.save_frames:
            for i in range(len(self.streams)):
                cam_dir = os.path.join(self.output_dir, f"camera_{i+1}")
                if not os.path.exists(cam_dir):
                    os.makedirs(cam_dir)
        
        # DeepStream pipeline elements
        self.pipeline = None
        self.loop = None
    
    def load_streams_config(self):
        """Load streams configuration from JSON"""
        try:
            with open(self.streams_config_file, 'r') as f:
                config = json.load(f)
                self.streams = config.get('streams', [])
                if not self.streams:
                    print("❌ No streams found in config")
                    sys.exit(1)
                print(f"✅ Loaded {len(self.streams)} streams")
        except Exception as e:
            print(f"❌ Error loading streams config: {e}")
            sys.exit(1)
    
    def load_polygon(self, stream_id):
        """Load polygon for specific stream"""
        polygon_file = self.stream_data[stream_id]['polygon_file']
        if os.path.exists(polygon_file):
            try:
                with open(polygon_file, 'r') as f:
                    data = json.load(f)
                    points = data.get('points', [])
                    if len(points) >= 3:
                        self.stream_data[stream_id]['polygon_points'] = points
                        self.stream_data[stream_id]['polygon_ready'] = True
                        print(f"✅ Loaded polygon for {self.stream_data[stream_id]['name']}")
            except Exception as e:
                print(f"❌ Error loading polygon for stream {stream_id}: {e}")
    
    def save_polygon(self, stream_id):
        """Save polygon for specific stream"""
        polygon_file = self.stream_data[stream_id]['polygon_file']
        try:
            with open(polygon_file, 'w') as f:
                json.dump({'points': self.stream_data[stream_id]['polygon_points']}, f, indent=2)
            print(f"✅ Saved polygon to {polygon_file}")
        except Exception as e:
            print(f"❌ Error saving polygon: {e}")
    
    def draw_polygon_ui(self, stream_id):
        """UI to draw polygon for specific stream"""
        stream = self.streams[stream_id]
        stream_name = self.stream_data[stream_id]['name']
        
        print(f"\n{'='*60}")
        print(f"POLYGON DRAWING MODE - {stream_name}")
        print(f"{'='*60}")
        print("Left click: Add point | Right click: Remove | 'c': Clear | 's': Save | 'q': Quit")
        
        # Capture first frame
        uri = stream['uri'].replace("file://", "")
        if uri.startswith("rtsp://"):
            print(f"⚠️  RTSP stream detected. Using placeholder for drawing.")
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(frame, f"{stream_name} - Draw polygon area", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        else:
            cap = cv2.VideoCapture(uri)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"❌ Error: Cannot read video from {uri}")
                return False
        
        original_frame = frame.copy()
        polygon_points = self.stream_data[stream_id]['polygon_points'].copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                polygon_points.append([x, y])
            elif event == cv2.EVENT_RBUTTONDOWN:
                if polygon_points:
                    polygon_points.pop()
        
        window_name = f'Draw Polygon - {stream_name}'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            display_frame = original_frame.copy()
            
            # Draw existing points and polygon
            if len(polygon_points) > 0:
                for i, point in enumerate(polygon_points):
                    cv2.circle(display_frame, tuple(point), 5, (0, 255, 0), -1)
                    cv2.putText(display_frame, str(i+1), tuple(point), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                if len(polygon_points) > 1:
                    pts = np.array(polygon_points, np.int32)
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                    
                    if len(polygon_points) >= 3:
                        overlay = display_frame.copy()
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
            
            # Show instructions
            cv2.putText(display_frame, f"{stream_name} - Points: {len(polygon_points)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 's' to save and continue", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if len(polygon_points) >= 3:
                    self.stream_data[stream_id]['polygon_points'] = polygon_points
                    self.stream_data[stream_id]['polygon_ready'] = True
                    self.save_polygon(stream_id)
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("⚠️  Need at least 3 points!")
            elif key == ord('c'):
                polygon_points = []
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
        
        return False
    
    def is_point_in_polygon(self, stream_id, x, y):
        """Check if point is inside polygon for specific stream"""
        data = self.stream_data[stream_id]
        if not data['polygon_ready'] or len(data['polygon_points']) < 3:
            return True
        
        try:
            point = Point(x, y)
            polygon = Polygon(data['polygon_points'])
            return polygon.contains(point)
        except:
            return True
    
    def save_frame_with_detections(self, stream_id, frame_array, objects_inside, frame_number):
        """Save frame with detection information"""
        try:
            frame_to_save = frame_array.copy()
            data = self.stream_data[stream_id]
            
            # Draw bbox if enabled
            if self.draw_bbox:
                # Draw polygon
                if data['polygon_ready'] and len(data['polygon_points']) >= 3:
                    pts = np.array(data['polygon_points'], np.int32)
                    cv2.polylines(frame_to_save, [pts], True, (0, 255, 0), 3)
                    overlay = frame_to_save.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.15, frame_to_save, 0.85, 0, frame_to_save)
                
                # Draw bboxes
                for obj in objects_inside:
                    x, y, w, h = obj['bbox']
                    cx, cy = obj['center']
                    
                    cv2.rectangle(frame_to_save, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame_to_save, (cx, cy), 5, (0, 0, 255), -1)
                    
                    label = f"{obj['class_name']} {obj['confidence']:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_to_save, (x, y - label_size[1] - 10), 
                                 (x + label_size[0], y), (0, 255, 0), -1)
                    cv2.putText(frame_to_save, label, (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Save frame
            data['saved_frame_count'] += 1
            cam_dir = os.path.join(self.output_dir, f"camera_{stream_id+1}")
            frame_filename = os.path.join(cam_dir, f"frame_{frame_number:06d}.jpg")
            cv2.imwrite(frame_filename, frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Save JSON
            json_filename = os.path.join(cam_dir, f"frame_{frame_number:06d}.json")
            detection_data = {
                'stream_id': stream_id,
                'stream_name': data['name'],
                'frame_number': frame_number,
                'num_objects': len(objects_inside),
                'polygon_points': data['polygon_points'],
                'objects': objects_inside
            }
            with open(json_filename, 'w') as f:
                json.dump(detection_data, f, indent=2)
            
        except Exception as e:
            print(f"Error saving frame {frame_number} for stream {stream_id}: {e}")
    
    def frame_saver_thread(self):
        """Background thread to save frames asynchronously"""
        while not self.stop_save_thread:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                
                if frame_data is None:
                    break
                
                stream_id, frame_array, objects_inside, frame_number = frame_data
                self.save_frame_with_detections(stream_id, frame_array, objects_inside, frame_number)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in saver thread: {e}")
    
    def start_frame_saver_thread(self):
        """Start background thread for saving frames"""
        if self.save_frames and self.save_thread is None:
            self.stop_save_thread = False
            self.save_thread = threading.Thread(target=self.frame_saver_thread, daemon=True)
            self.save_thread.start()
    
    def stop_frame_saver_thread(self):
        """Stop background thread for saving frames"""
        if self.save_thread is not None:
            self.stop_save_thread = True
            self.frame_queue.put(None)
            self.save_thread.join(timeout=5)
    
    def tiler_src_pad_buffer_probe(self, pad, info, u_data):
        """Probe function after tiler (on src pad) to draw polygons and text on tiled output"""
        if not pyds:
            return Gst.PadProbeReturn.OK
            
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        
        num_streams = len(self.streams)
        tiler_rows = int(np.ceil(np.sqrt(num_streams)))
        tiler_cols = int(np.ceil(num_streams / tiler_rows))
        
        tile_width = 1920 // tiler_cols
        tile_height = 1080 // tiler_rows
        
        # After tiler, there's only ONE frame (the tiled composite)
        l_frame = batch_meta.frame_meta_list
        if l_frame is None:
            return Gst.PadProbeReturn.OK
        
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            return Gst.PadProbeReturn.OK
        
        # Create ONE display meta for the entire tiled frame
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        
        # Draw polygon and text for EACH camera position in the tiled view
        for stream_id in range(num_streams):
            if stream_id not in self.stream_data:
                continue
                
            data = self.stream_data[stream_id]
            
            # Calculate tile position for this stream
            tile_row = stream_id // tiler_cols
            tile_col = stream_id % tiler_cols
            
            offset_x = tile_col * tile_width
            offset_y = tile_row * tile_height
            
            # Calculate scale (original frames are 1920x1080, tiles are smaller)
            scale_x = tile_width / 1920.0
            scale_y = tile_height / 1080.0
            
            # Draw polygon for this stream
            if data['polygon_ready'] and len(data['polygon_points']) >= 3:
                for i in range(len(data['polygon_points'])):
                    if display_meta.num_lines >= 16:
                        break
                        
                    p1 = data['polygon_points'][i]
                    p2 = data['polygon_points'][(i + 1) % len(data['polygon_points'])]
                    
                    line_params = display_meta.line_params[display_meta.num_lines]
                    line_params.x1 = int(p1[0] * scale_x + offset_x)
                    line_params.y1 = int(p1[1] * scale_y + offset_y)
                    line_params.x2 = int(p2[0] * scale_x + offset_x)
                    line_params.y2 = int(p2[1] * scale_y + offset_y)
                    line_params.line_width = 3
                    line_params.line_color.set(0.0, 1.0, 0.0, 1.0)
                    display_meta.num_lines += 1
            
            # Draw text (camera name + FPS) for this stream
            if display_meta.num_labels < 16:
                text_params = display_meta.text_params[display_meta.num_labels]
                text_params.display_text = f"{data['name']} | FPS: {data['fps']:.2f}"
                text_params.x_offset = offset_x + 10
                text_params.y_offset = offset_y + 30
                text_params.font_params.font_name = "Serif"
                text_params.font_params.font_size = 14
                text_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)
                text_params.set_bg_clr = 1
                text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
                display_meta.num_labels += 1
        
        # Add display meta to the tiled frame
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
        return Gst.PadProbeReturn.OK
    
    def pgie_src_pad_buffer_probe(self, pad, info, u_data):
        """Probe function after PGIE (BEFORE tiler) to filter detections by polygon"""
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
            
            # Get stream ID (source_id from frame_meta)
            stream_id = frame_meta.source_id
            
            if stream_id not in self.stream_data:
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue
            
            data = self.stream_data[stream_id]
            data['frame_count'] += 1
            
            # Calculate FPS
            if data['fps_start_time'] is None:
                data['fps_start_time'] = time.time()
                data['fps_frame_count'] = 0
            
            data['fps_frame_count'] += 1
            
            if data['fps_frame_count'] >= 30:
                elapsed_time = time.time() - data['fps_start_time']
                if elapsed_time > 0:
                    data['fps'] = data['fps_frame_count'] / elapsed_time
                data['fps_start_time'] = time.time()
                data['fps_frame_count'] = 0
            
            # Filter objects by polygon (no display meta here, will be added after tiler)
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
                
                # Calculate center point
                center_x = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
                center_y = obj_meta.rect_params.top + obj_meta.rect_params.height / 2
                
                # Check if in polygon
                if not self.is_point_in_polygon(stream_id, center_x, center_y):
                    objects_to_remove.append(obj_meta)
                else:
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
            
            # Console logging
            if len(objects_inside) > 0:
                print(f"[{data['name']:12s}] Frame {data['frame_count']:5d} | FPS: {data['fps']:6.2f} | Objects: {len(objects_inside)}/{num_total}")
                
                # Save frame if enabled
                if self.save_frames:
                    try:
                        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                        frame_height = frame_meta.source_frame_height
                        frame_width = frame_meta.source_frame_width
                        
                        frame_rgba = np.array(n_frame, copy=True, order='C')
                        frame_rgba = frame_rgba.reshape((frame_height, frame_width, 4))
                        frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
                        
                        try:
                            objects_copy = [obj.copy() for obj in objects_inside]
                            self.frame_queue.put((stream_id, frame_bgr, objects_copy, data['frame_count']), block=False)
                        except queue.Full:
                            pass
                    except Exception as e:
                        if data['frame_count'] % 100 == 1:
                            print(f"Error extracting frame: {type(e).__name__}")
            
            elif data['frame_count'] % 100 == 0:
                print(f"[{data['name']:12s}] Frame {data['frame_count']:5d} | FPS: {data['fps']:6.2f} | No objects ({num_total} detected)")
            
            # Remove objects outside polygon
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
    
    def bus_call(self, bus, message, loop):
        """Handle bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"Warning: {err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}: {debug}")
            loop.quit()
        return True
    
    def cb_newpad(self, decodebin, decoder_src_pad, data):
        """Callback for new pad from uridecodebin"""
        caps = decoder_src_pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)
        
        if gstname.find("video") != -1:
            if features.contains("memory:NVMM"):
                bin_ghost_pad = source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
            else:
                sys.stderr.write("Error: Decodebin did not pick nvidia decoder plugin.\n")
    
    def decodebin_child_added(self, child_proxy, Object, name, user_data):
        """Callback for decodebin child added"""
        if name.find("decodebin") != -1:
            Object.connect("child-added", self.decodebin_child_added, user_data)
    
    def create_source_bin(self, index, uri):
        """Create source bin for uridecodebin"""
        bin_name = "source-bin-%02d" % index
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            sys.stderr.write("Unable to create source bin\n")
            return None
        
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            sys.stderr.write("Unable to create uri decode bin\n")
            return None
        
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", self.cb_newpad, nbin)
        uri_decode_bin.connect("child-added", self.decodebin_child_added, nbin)
        
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write("Failed to add ghost pad in source bin\n")
            return None
        
        return nbin
    
    def create_pipeline(self):
        """Create GStreamer pipeline"""
        Gst.init(None)
        
        self.pipeline = Gst.Pipeline()
        
        if not self.pipeline:
            print("Unable to create Pipeline")
            return False
        
        # Streammux
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batch-size', len(self.streams))
        streammux.set_property('batched-push-timeout', 40000)
        
        # PGIE
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        pgie.set_property('config-file-path', self.config_file)
        
        # Tiler for multi-stream display
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        tiler_rows = int(np.ceil(np.sqrt(len(self.streams))))
        tiler_cols = int(np.ceil(len(self.streams) / tiler_rows))
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_cols)
        tiler.set_property("width", 1920)
        tiler.set_property("height", 1080)
        
        # Video converter
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        
        # Caps filter for RGBA
        caps_rgba = Gst.ElementFactory.make("capsfilter", "caps-rgba")
        caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        
        # OSD
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        
        # Tee
        tee = Gst.ElementFactory.make("tee", "nvsink-tee")
        
        # Display branch
        queue_display = Gst.ElementFactory.make("queue", "nvtee-que-display")
        nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
        caps_display = Gst.ElementFactory.make("capsfilter", "filter-display")
        caps_display.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        sink_display = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        sink_display.set_property('sync', False)
        
        # RTSP branch
        queue_rtsp = Gst.ElementFactory.make("queue", "nvtee-que-rtsp")
        nvvidconv_rtsp = Gst.ElementFactory.make("nvvideoconvert", "convertor-rtsp")
        caps_rtsp = Gst.ElementFactory.make("capsfilter", "filter-rtsp")
        caps_rtsp.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
        
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        if not encoder:
            encoder = Gst.ElementFactory.make("x264enc", "encoder")
            if encoder:
                encoder.set_property("bitrate", 5000)
        
        if not encoder:
            print("❌ Unable to create encoder")
            return False
        
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        sink_rtsp = Gst.ElementFactory.make("udpsink", "udpsink")
        sink_rtsp.set_property("host", "127.0.0.1")
        sink_rtsp.set_property("port", self.udp_port_start)
        sink_rtsp.set_property("async", False)
        sink_rtsp.set_property("sync", 0)
        
        # Check all elements
        if not all([streammux, pgie, tiler, nvvidconv, caps_rgba, nvosd, tee,
                    queue_display, nvvidconv_postosd, caps_display, transform, sink_display,
                    queue_rtsp, nvvidconv_rtsp, caps_rtsp, encoder, rtppay, sink_rtsp]):
            print("❌ Unable to create one or more elements")
            return False
        
        # Add elements to pipeline
        self.pipeline.add(streammux)
        self.pipeline.add(pgie)
        self.pipeline.add(tiler)
        self.pipeline.add(nvvidconv)
        self.pipeline.add(caps_rgba)
        self.pipeline.add(nvosd)
        self.pipeline.add(tee)
        
        # Display branch
        self.pipeline.add(queue_display)
        self.pipeline.add(nvvidconv_postosd)
        self.pipeline.add(caps_display)
        self.pipeline.add(transform)
        self.pipeline.add(sink_display)
        
        # RTSP branch
        self.pipeline.add(queue_rtsp)
        self.pipeline.add(nvvidconv_rtsp)
        self.pipeline.add(caps_rtsp)
        self.pipeline.add(encoder)
        self.pipeline.add(rtppay)
        self.pipeline.add(sink_rtsp)
        
        # Create and add source bins
        for i, stream in enumerate(self.streams):
            source_bin = self.create_source_bin(i, stream['uri'])
            if not source_bin:
                return False
            
            self.pipeline.add(source_bin)
            
            # Link to streammux
            srcpad = source_bin.get_static_pad("src")
            sinkpad = streammux.get_request_pad(f"sink_{i}")
            srcpad.link(sinkpad)
        
        # Link main pipeline
        streammux.link(pgie)
        pgie.link(nvvidconv)
        nvvidconv.link(caps_rgba)
        caps_rgba.link(tiler)
        tiler.link(nvosd)
        nvosd.link(tee)
        
        # Link display branch
        tee_display_pad = tee.get_request_pad("src_0")
        queue_display_pad = queue_display.get_static_pad("sink")
        tee_display_pad.link(queue_display_pad)
        
        queue_display.link(nvvidconv_postosd)
        nvvidconv_postosd.link(caps_display)
        caps_display.link(transform)
        transform.link(sink_display)
        
        # Link RTSP branch
        tee_rtsp_pad = tee.get_request_pad("src_1")
        queue_rtsp_pad = queue_rtsp.get_static_pad("sink")
        tee_rtsp_pad.link(queue_rtsp_pad)
        
        queue_rtsp.link(nvvidconv_rtsp)
        nvvidconv_rtsp.link(caps_rtsp)
        caps_rtsp.link(encoder)
        encoder.link(rtppay)
        rtppay.link(sink_rtsp)
        
        # Add probe to PGIE src pad (BEFORE tiling, for filtering objects by polygon)
        if pyds:
            pgie_src_pad = pgie.get_static_pad("src")
            if not pgie_src_pad:
                print("Unable to get src pad of pgie")
            else:
                pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.pgie_src_pad_buffer_probe, 0)
        
        # Add probe to tiler src pad (AFTER tiling, for drawing polygons on tiled output)
        if pyds:
            tiler_src_pad = tiler.get_static_pad("src")
            if tiler_src_pad:
                tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.tiler_src_pad_buffer_probe, 0)
        
        return True
    
    def run(self):
        """Run the application"""
        # Step 1: Draw polygons for each stream
        for i in range(len(self.streams)):
            data = self.stream_data[i]
            
            if not data['polygon_ready']:
                if not self.draw_polygon_ui(i):
                    return
            else:
                if not self.skip_prompt:
                    response = input(f"Redraw polygon for {data['name']}? (y/n): ").strip().lower()
                    if response == 'y':
                        if not self.draw_polygon_ui(i):
                            return
        
        # Step 2: Create and run pipeline
        if not self.create_pipeline():
            return
        
        # Step 3: Start RTSP server
        server = GstRtspServer.RTSPServer.new()
        server.props.service = "%d" % self.rtsp_port
        server.attach(None)
        
        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(
            '( udpsrc name=pay0 port=%d buffer-size=524288 '
            'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)H264, payload=96" )'
            % self.udp_port_start
        )
        factory.set_shared(True)
        server.get_mount_points().add_factory("/multi-camera", factory)
        
        # Create event loop
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        print("\n" + "="*60)
        print("DeepStream Multi-Camera Polygon Detection Started")
        print("="*60)
        print(f"Number of cameras: {len(self.streams)}")
        print(f"RTSP URL: rtsp://localhost:{self.rtsp_port}/multi-camera")
        if self.save_frames:
            print(f"Saving frames to: {self.output_dir}")
        print("Press Ctrl+C to stop\n")
        
        # Start frame saver thread
        if self.save_frames:
            self.start_frame_saver_thread()
        
        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        
        # Cleanup
        self.pipeline.set_state(Gst.State.NULL)
        
        if self.save_frames:
            self.stop_frame_saver_thread()
            total_saved = sum(data['saved_frame_count'] for data in self.stream_data.values())
            print(f"\nSaved {total_saved} frames total to {self.output_dir}")
        
        print("Done")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepStream Multi-Camera Polygon Detection App')
    parser.add_argument('-s', '--streams', required=True,
                       help='Streams configuration JSON file')
    parser.add_argument('-c', '--config', required=True,
                       help='Primary inference config file')
    parser.add_argument('--rtsp-port', type=int, default=8554,
                       help='RTSP server port (default: 8554)')
    parser.add_argument('--udp-port-start', type=int, default=5400,
                       help='UDP sink port start (default: 5400)')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save frames with objects in polygon')
    parser.add_argument('--output-dir', default='output_frames',
                       help='Directory to save frames (default: output_frames)')
    parser.add_argument('--no-draw-bbox', action='store_true',
                       help='Do not draw bounding boxes on saved frames')
    parser.add_argument('--skip-prompt', action='store_true',
                       help='Skip polygon redraw prompt')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.streams):
        print(f"Error: Streams config not found: {args.streams}")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    # Create and run app
    app = MultiCameraPolygonApp(args.streams, args.config,
                                args.rtsp_port, args.udp_port_start,
                                args.save_frames, args.output_dir,
                                not args.no_draw_bbox, args.skip_prompt)
    app.run()


if __name__ == '__main__':
    main()
