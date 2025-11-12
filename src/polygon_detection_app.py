#!/usr/bin/env python3

"""
DeepStream Polygon Detection App
Vẽ polygon trên frame và chỉ detect object trong vùng polygon
"""

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

try:
    import pyds
except ImportError:
    print("WARNING: pyds module not found. Some features may not work.")
    pyds = None

from shapely.geometry import Point, Polygon


class PolygonDetectionApp:
    def __init__(self, video_source, config_file, polygon_file="polygon.json", 
                 rtsp_port=8554, udp_port=5400, output_file=None, 
                 save_frames=False, output_dir="output_frames", draw_bbox=True,
                 skip_prompt=False):
        self.video_source = video_source
        self.config_file = config_file
        self.polygon_file = polygon_file
        self.polygon_points = []
        self.drawing = False
        self.frame_for_drawing = None
        self.polygon_ready = False
        self.rtsp_port = rtsp_port
        self.udp_port = udp_port
        self.output_file = output_file
        self.frame_count = 0
        
        # FPS calculation
        self.fps = 0.0
        self.fps_start_time = None
        self.fps_frame_count = 0
        self.fps_update_interval = 30  # Update FPS every 30 frames
        
        # Frame saving options
        self.save_frames = save_frames
        self.output_dir = output_dir
        self.draw_bbox = draw_bbox
        self.saved_frame_count = 0
        self.skip_prompt = skip_prompt
        
        # Create output directory if needed
        if self.save_frames and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Threading for async frame saving
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer max 30 frames
        self.save_thread = None
        self.stop_save_thread = False
        
        # Load polygon if exists
        self.load_polygon()
        
        # DeepStream pipeline elements
        self.pipeline = None
        self.loop = None
        
    def load_polygon(self):
        """Load polygon from JSON file"""
        if os.path.exists(self.polygon_file):
            try:
                with open(self.polygon_file, 'r') as f:
                    data = json.load(f)
                    self.polygon_points = data.get('points', [])
                    if len(self.polygon_points) >= 3:
                        self.polygon_ready = True
            except Exception as e:
                print(f"❌ Error loading polygon: {e}")
    
    def save_polygon(self):
        """Save polygon to JSON file"""
        try:
            with open(self.polygon_file, 'w') as f:
                json.dump({'points': self.polygon_points}, f, indent=2)
        except Exception as e:
            print(f"Error saving polygon: {e}")
    
    def draw_polygon_ui(self):
        """UI to draw polygon on first frame"""
        print("\nPOLYGON DRAWING MODE")
        print("Left click: Add point | Right click: Remove | 'c': Clear | 's': Save | 'q': Quit")
        
        # Capture first frame
        cap = cv2.VideoCapture(self.video_source.replace("file://", ""))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Cannot read video")
            return False
        
        self.frame_for_drawing = frame.copy()
        original_frame = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.polygon_points.append([x, y])
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.polygon_points:
                    self.polygon_points.pop()
        
        cv2.namedWindow('Draw Polygon')
        cv2.setMouseCallback('Draw Polygon', mouse_callback)
        
        while True:
            display_frame = original_frame.copy()
            
            # Draw existing points and polygon
            if len(self.polygon_points) > 0:
                for i, point in enumerate(self.polygon_points):
                    cv2.circle(display_frame, tuple(point), 5, (0, 255, 0), -1)
                    cv2.putText(display_frame, str(i+1), tuple(point), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                if len(self.polygon_points) > 1:
                    pts = np.array(self.polygon_points, np.int32)
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                    
                    if len(self.polygon_points) >= 3:
                        # Fill polygon with transparency
                        overlay = display_frame.copy()
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
            
            # Show instructions
            cv2.putText(display_frame, f"Points: {len(self.polygon_points)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 's' to save and start", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Draw Polygon', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if len(self.polygon_points) >= 3:
                    self.polygon_ready = True
                    self.save_polygon()
                    cv2.destroyAllWindows()
                    return True
            elif key == ord('c'):
                self.polygon_points = []
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
        
        return False
    
    def is_point_in_polygon(self, x, y):
        """Check if point is inside polygon"""
        if not self.polygon_ready or len(self.polygon_points) < 3:
            return True  # If no polygon, detect everything
        
        try:
            point = Point(x, y)
            polygon = Polygon(self.polygon_points)
            return polygon.contains(point)
        except:
            return True
    
    def save_frame_with_detections(self, frame_array, objects_inside, frame_number):
        """Save frame with detection information"""
        try:
            # Create frame copy
            frame_to_save = frame_array.copy()
            
            # Draw bbox if enabled
            if self.draw_bbox:
                # Draw polygon first
                if self.polygon_ready and len(self.polygon_points) >= 3:
                    pts = np.array(self.polygon_points, np.int32)
                    cv2.polylines(frame_to_save, [pts], True, (0, 255, 0), 3)
                    overlay = frame_to_save.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.15, frame_to_save, 0.85, 0, frame_to_save)
                
                # Draw bboxes for objects inside polygon
                for obj in objects_inside:
                    x, y, w, h = obj['bbox']
                    cx, cy = obj['center']
                    
                    # Draw bounding box
                    cv2.rectangle(frame_to_save, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw center point
                    cv2.circle(frame_to_save, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Draw label
                    label = f"{obj['class_name']} {obj['confidence']:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_to_save, (x, y - label_size[1] - 10), 
                                 (x + label_size[0], y), (0, 255, 0), -1)
                    cv2.putText(frame_to_save, label, (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Save frame
            self.saved_frame_count += 1
            frame_filename = os.path.join(self.output_dir, f"frame_{frame_number:06d}.jpg")
            cv2.imwrite(frame_filename, frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Save detection info to JSON
            json_filename = os.path.join(self.output_dir, f"frame_{frame_number:06d}.json")
            detection_data = {
                'frame_number': frame_number,
                'num_objects': len(objects_inside),
                'polygon_points': self.polygon_points,
                'objects': objects_inside
            }
            with open(json_filename, 'w') as f:
                json.dump(detection_data, f, indent=2)
            
        except Exception as e:
            print(f"Error saving frame {frame_number}: {e}")
    
    def frame_saver_thread(self):
        """Background thread to save frames asynchronously"""
        while not self.stop_save_thread:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                
                if frame_data is None:
                    break
                
                frame_array, objects_inside, frame_number = frame_data
                self.save_frame_with_detections(frame_array, objects_inside, frame_number)
                
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
    
    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        """Probe function to filter detections by polygon"""
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
            
            self.frame_count += 1
            
            # Calculate FPS
            if self.fps_start_time is None:
                self.fps_start_time = time.time()
                self.fps_frame_count = 0
            
            self.fps_frame_count += 1
            
            if self.fps_frame_count >= self.fps_update_interval:
                elapsed_time = time.time() - self.fps_start_time
                if elapsed_time > 0:
                    self.fps = self.fps_frame_count / elapsed_time
                self.fps_start_time = time.time()
                self.fps_frame_count = 0
            
            # Draw polygon on frame
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            
            if self.polygon_ready and len(self.polygon_points) >= 3:
                # Draw polygon lines
                for i in range(len(self.polygon_points)):
                    p1 = self.polygon_points[i]
                    p2 = self.polygon_points[(i + 1) % len(self.polygon_points)]
                    
                    line_params = display_meta.line_params[i]
                    line_params.x1 = p1[0]
                    line_params.y1 = p1[1]
                    line_params.x2 = p2[0]
                    line_params.y2 = p2[1]
                    line_params.line_width = 3
                    line_params.line_color.set(0.0, 1.0, 0.0, 1.0)
                    display_meta.num_lines += 1
                    
                    if display_meta.num_lines >= 16:  # Max lines limit
                        break
            
            # Display FPS on frame
            if self.fps > 0:
                fps_text = f"FPS: {self.fps:.2f}"
                text_params = display_meta.text_params[0]
                text_params.display_text = fps_text
                text_params.x_offset = 10
                text_params.y_offset = 30
                text_params.font_params.font_name = "Serif"
                text_params.font_params.font_size = 14
                text_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)  # Yellow
                text_params.set_bg_clr = 1
                text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)  # Black background
                display_meta.num_labels += 1
            
            # Filter objects by polygon
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
                
                # Calculate center point of bounding box
                center_x = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
                center_y = obj_meta.rect_params.top + obj_meta.rect_params.height / 2
                
                # Check if center is in polygon
                if not self.is_point_in_polygon(center_x, center_y):
                    # Mark for removal (outside polygon)
                    objects_to_remove.append(obj_meta)
                else:
                    # Inside polygon - keep and highlight
                    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
                    obj_meta.rect_params.border_width = 3
                    
                    # Get class name
                    class_name = obj_meta.obj_label if obj_meta.obj_label else f"class_{obj_meta.class_id}"
                    
                    # Save for logging
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
                print(f"[Frame {self.frame_count:5d}] FPS: {self.fps:6.2f} | Objects: {len(objects_inside)}/{num_total}")
                
                # Save frame if enabled (async via queue)
                if self.save_frames:
                    try:
                        # Get frame data from NvBufSurface (must be RGBA format from caps_rgba)
                        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                        
                        # Get frame dimensions
                        frame_height = frame_meta.source_frame_height
                        frame_width = frame_meta.source_frame_width
                        
                        # Convert to numpy array and reshape (RGBA has 4 channels)
                        frame_rgba = np.array(n_frame, copy=True, order='C')
                        frame_rgba = frame_rgba.reshape((frame_height, frame_width, 4))
                        
                        # Convert RGBA to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
                        
                        # Add to queue (non-blocking)
                        try:
                            # Make a copy of objects_inside to avoid threading issues
                            objects_copy = [obj.copy() for obj in objects_inside]
                            self.frame_queue.put((frame_bgr, objects_copy, self.frame_count), block=False)
                        
                        except queue.Full:
                            pass
                    
                    except RuntimeError as e:
                        if "color Format" in str(e) and self.frame_count == 1:
                            print(f"ERROR: Frame format is not RGBA!")
                    
                    except Exception as e:
                        if self.frame_count % 100 == 1:
                            print(f"Error: {type(e).__name__}")

            
            # Print summary every 30 frames
            elif self.frame_count % 100 == 0:
                print(f"[Frame {self.frame_count:5d}] FPS: {self.fps:6.2f} | No objects ({num_total} detected)")
            
            # Remove objects outside polygon
            for obj_meta in objects_to_remove:
                try:
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                except:
                    pass
            
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            
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
        
        # Create source bin
        source_bin = self.create_source_bin(0, self.video_source)
        if not source_bin:
            return False
        
        # Streammux
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', 40000)
        
        # PGIE (Primary inference)
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        pgie.set_property('config-file-path', self.config_file)
        
        # Video converter (to RGBA for frame extraction)
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        
        # Caps filter to force RGBA format (required for pyds.get_nvds_buf_surface)
        caps_rgba = Gst.ElementFactory.make("capsfilter", "caps-rgba")
        caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        
        # OSD
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        
        # Tee to split stream
        tee = Gst.ElementFactory.make("tee", "nvsink-tee")
        
        # Queue for display branch
        queue_display = Gst.ElementFactory.make("queue", "nvtee-que-display")
        
        # Video converter for display
        nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
        
        # Caps filter for display - IMPORTANT: Set to RGBA for pyds.get_nvds_buf_surface()
        caps_display = Gst.ElementFactory.make("capsfilter", "filter-display")
        caps_display.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        
        # EGL transform
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        
        # Display sink
        sink_display = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        sink_display.set_property('sync', False)
        
        # Queue for file output branch (if output file specified)
        queue_file = None
        nvvidconv_file = None
        caps_file = None
        encoder_file = None
        parser_file = None
        mux_file = None
        sink_file = None
        
        if self.output_file:
            queue_file = Gst.ElementFactory.make("queue", "nvtee-que-file")
            nvvidconv_file = Gst.ElementFactory.make("nvvideoconvert", "convertor-file")
            caps_file = Gst.ElementFactory.make("capsfilter", "filter-file")
            caps_file.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
            
            encoder_file = Gst.ElementFactory.make("nvv4l2h264enc", "encoder-file")
            if not encoder_file:
                encoder_file = Gst.ElementFactory.make("x264enc", "encoder-file")
            
            parser_file = Gst.ElementFactory.make("h264parse", "h264-parser")
            mux_file = Gst.ElementFactory.make("mp4mux", "mp4-mux")
            sink_file = Gst.ElementFactory.make("filesink", "file-sink")
            sink_file.set_property("location", self.output_file)
            sink_file.set_property("sync", False)
        
        # Queue for RTSP branch
        queue_rtsp = Gst.ElementFactory.make("queue", "nvtee-que-rtsp")
        
        # Video converter for RTSP
        nvvidconv_rtsp = Gst.ElementFactory.make("nvvideoconvert", "convertor-rtsp")
        
        # Caps filter for RTSP
        caps_rtsp = Gst.ElementFactory.make("capsfilter", "filter-rtsp")
        caps_rtsp.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
        
        # Encoder
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        if not encoder:
            encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        if not encoder:
            encoder = Gst.ElementFactory.make("x264enc", "encoder")
            encoder.set_property("bitrate", 5000000)
        
        # RTP payload
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        
        # UDP sink for RTSP
        sink_rtsp = Gst.ElementFactory.make("udpsink", "udpsink")
        sink_rtsp.set_property("host", "127.0.0.1")
        sink_rtsp.set_property("port", self.udp_port)
        sink_rtsp.set_property("async", False)
        sink_rtsp.set_property("sync", 0)
        
        if not streammux or not pgie or not nvvidconv or not caps_rgba or not nvosd or not tee:
            print("Unable to create elements")
            return False
        
        # Add elements to pipeline
        self.pipeline.add(source_bin)
        self.pipeline.add(streammux)
        self.pipeline.add(pgie)
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
        
        # File output branch (if enabled)
        if self.output_file and queue_file:
            self.pipeline.add(queue_file)
            self.pipeline.add(nvvidconv_file)
            self.pipeline.add(caps_file)
            self.pipeline.add(encoder_file)
            self.pipeline.add(parser_file)
            self.pipeline.add(mux_file)
            self.pipeline.add(sink_file)
        
        # RTSP branch
        self.pipeline.add(queue_rtsp)
        self.pipeline.add(nvvidconv_rtsp)
        self.pipeline.add(caps_rtsp)
        self.pipeline.add(encoder)
        self.pipeline.add(rtppay)
        self.pipeline.add(sink_rtsp)
        
        # Link source bin to streammux
        srcpad = source_bin.get_static_pad("src")
        sinkpad = streammux.get_request_pad("sink_0")
        srcpad.link(sinkpad)
        
        # Link main pipeline up to tee
        streammux.link(pgie)
        pgie.link(nvvidconv)
        nvvidconv.link(caps_rgba)
        caps_rgba.link(nvosd)
        nvosd.link(tee)
        
        # Link display branch
        tee_display_pad = tee.get_request_pad("src_0")
        queue_display_pad = queue_display.get_static_pad("sink")
        tee_display_pad.link(queue_display_pad)
        
        queue_display.link(nvvidconv_postosd)
        nvvidconv_postosd.link(caps_display)
        caps_display.link(transform)
        transform.link(sink_display)
        
        # Link file output branch (if enabled)
        src_pad_idx = 1
        if self.output_file and queue_file:
            tee_file_pad = tee.get_request_pad(f"src_{src_pad_idx}")
            queue_file_pad = queue_file.get_static_pad("sink")
            tee_file_pad.link(queue_file_pad)
            
            queue_file.link(nvvidconv_file)
            nvvidconv_file.link(caps_file)
            caps_file.link(encoder_file)
            encoder_file.link(parser_file)
            parser_file.link(mux_file)
            mux_file.link(sink_file)
            
            src_pad_idx += 1
        
        # Link RTSP branch
        tee_rtsp_pad = tee.get_request_pad(f"src_{src_pad_idx}")
        queue_rtsp_pad = queue_rtsp.get_static_pad("sink")
        tee_rtsp_pad.link(queue_rtsp_pad)
        
        queue_rtsp.link(nvvidconv_rtsp)
        nvvidconv_rtsp.link(caps_rtsp)
        caps_rtsp.link(encoder)
        encoder.link(rtppay)
        rtppay.link(sink_rtsp)
        
        # Add probe to OSD sink pad
        if pyds:
            osdsinkpad = nvosd.get_static_pad("sink")
            if not osdsinkpad:
                print("Unable to get sink pad of nvosd")
            else:
                osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)
        
        return True
    
    def run(self):
        """Run the application"""
        # Step 1: Draw polygon if not exists or user wants to redraw
        if not self.polygon_ready:
            if not self.draw_polygon_ui():
                return
        else:
            if not self.skip_prompt:
                response = input("Redraw polygon? (y/n): ").strip().lower()
                if response == 'y':
                    if not self.draw_polygon_ui():
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
            % self.udp_port
        )
        factory.set_shared(True)
        server.get_mount_points().add_factory("/polygon-detect", factory)
        
        # Create event loop
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        print("\nDeepStream Polygon Detection Started")
        print(f"Video: {self.video_source}")
        print(f"RTSP: rtsp://localhost:{self.rtsp_port}/polygon-detect")
        if self.save_frames:
            print(f"Saving frames to: {self.output_dir}")
        print("Press Ctrl+C to stop\n")
        
        # Start frame saver thread if enabled
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
            print(f"\nSaved {self.saved_frame_count} frames to {self.output_dir}")
        
        print("Done")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepStream Polygon Detection App')
    parser.add_argument('-i', '--input', required=True, 
                       help='Input video file path')
    parser.add_argument('-c', '--config', required=True,
                       help='Primary inference config file')
    parser.add_argument('-p', '--polygon', default='polygon.json',
                       help='Polygon JSON file (default: polygon.json)')
    parser.add_argument('--rtsp-port', type=int, default=8554,
                       help='RTSP server port (default: 8554)')
    parser.add_argument('--udp-port', type=int, default=5400,
                       help='UDP sink port (default: 5400)')
    parser.add_argument('-o', '--output', default=None,
                       help='Output video file (e.g., output.mp4)')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save frames with objects in polygon')
    parser.add_argument('--output-dir', default='output_frames',
                       help='Directory to save frames (default: output_frames)')
    parser.add_argument('--no-draw-bbox', action='store_true',
                       help='Do not draw bounding boxes on saved frames')
    parser.add_argument('--skip-prompt', action='store_true',
                       help='Skip polygon redraw prompt (use existing polygon)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.input):
        print(f"Error: Video file not found: {args.input}")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    # Convert to file URI if needed
    video_source = args.input if args.input.startswith('file://') else f"file://{os.path.abspath(args.input)}"
    
    # Create and run app
    app = PolygonDetectionApp(video_source, args.config, args.polygon, 
                              args.rtsp_port, args.udp_port, args.output,
                              args.save_frames, args.output_dir, not args.no_draw_bbox,
                              args.skip_prompt)
    app.run()


if __name__ == '__main__':
    main()
