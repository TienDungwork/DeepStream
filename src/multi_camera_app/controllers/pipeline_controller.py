import sys
import gi
import numpy as np

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer

try:
    import pyds
except ImportError:
    print("WARNING: pyds module not found")
    pyds = None

from typing import Dict
from multi_camera_app.models import StreamData, AppConfig
from multi_camera_app.utils import PolygonUtils
from multi_camera_app.threads import FrameSaverThread, RTMPStreamManager
from multi_camera_app.controllers.gstreamer_builder import GStreamerBuilder
from multi_camera_app.controllers.probe_handler import ProbeHandler


class PipelineController:
    def __init__(self, config: AppConfig, streams: Dict[int, StreamData]):
        self.config = config
        self.streams = streams
        self.pipeline = None
        self.loop = None
        self.polygon_utils = PolygonUtils()
        self.frame_saver = None
        self.rtmp_stream_manager = None
        self.probe_handler = None
        self.builder = GStreamerBuilder()
        
        if config.save_frames:
            self.frame_saver = FrameSaverThread(config.output_dir, config.draw_bbox)
        
        if config.enable_rtmp_streaming:
            self.rtmp_stream_manager = RTMPStreamManager(
                rtmp_server_url=config.rtmp_server_url,
                stream_width=config.rtmp_stream_width,
                stream_height=config.rtmp_stream_height,
                fps=config.rtmp_fps,
                bitrate=config.rtmp_bitrate
            )
            
            for stream_id, stream_data in streams.items():
                self.rtmp_stream_manager.create_streamer(stream_id, stream_data.name)
                print(f"RTMP: {stream_data.name} -> {self.rtmp_stream_manager.get_stream_url(stream_id)}")
        
        self.probe_handler = ProbeHandler(
            streams, 
            self.polygon_utils, 
            self.frame_saver, 
            config.save_frames, 
            config.draw_bbox,
            config.streammux_width,
            config.streammux_height,
            self.rtmp_stream_manager
        )
    
    def create_pipeline(self):
        Gst.init(None)
        
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            print("Unable to create Pipeline")
            return False
        
        streammux = self.builder.create_streammux(
            self.config.streammux_width,
            self.config.streammux_height,
            self.config.streammux_batch_size,
            self.config.streammux_batched_push_timeout
        )
        
        pgie = self.builder.create_pgie(self.config.inference_config_file)
        
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvvidconv.set_property("nvbuf-memory-type", 0)
        
        caps_rgba = Gst.ElementFactory.make("capsfilter", "caps-rgba")
        caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        
        tiler_rows = int(np.ceil(np.sqrt(len(self.streams))))
        tiler_cols = int(np.ceil(len(self.streams) / tiler_rows))
        tiler = self.builder.create_tiler(tiler_rows, tiler_cols, 
                                          self.config.tiler_width, self.config.tiler_height)
        
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        
        # Tạo elements dựa trên chế độ headless
        if not self.config.headless:
            tee = Gst.ElementFactory.make("tee", "nvsink-tee")
            
            queue_display = Gst.ElementFactory.make("queue", "nvtee-que-display")
            queue_display.set_property("max-size-buffers", 10)
            queue_display.set_property("max-size-time", 0)
            queue_display.set_property("max-size-bytes", 0)
            
            nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
            nvvidconv_postosd.set_property("nvbuf-memory-type", 0)
            caps_display = Gst.ElementFactory.make("capsfilter", "filter-display")
            caps_display.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
            sink_display = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            sink_display.set_property('sync', False)
            sink_display.set_property('async', False)
            sink_display.set_property('max-lateness', -1)
        else:
            # Chế độ headless: không cần tee và display elements
            tee = None
            queue_display = None
            nvvidconv_postosd = None
            caps_display = None
            transform = None
            sink_display = None
        
        queue_rtsp = Gst.ElementFactory.make("queue", "nvtee-que-rtsp")
        queue_rtsp.set_property("max-size-buffers", 5)
        queue_rtsp.set_property("max-size-time", 0)
        queue_rtsp.set_property("max-size-bytes", 0)
        queue_rtsp.set_property("leaky", 2)
        queue_rtsp.set_property("flush-on-eos", True)
        
        nvvidconv_rtsp = Gst.ElementFactory.make("nvvideoconvert", "convertor-rtsp")
        nvvidconv_rtsp.set_property("nvbuf-memory-type", 0)

        caps_rtsp = Gst.ElementFactory.make("capsfilter", "caps-rtsp")
        caps_rtsp.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
        
        encoder = self.builder.create_encoder()
        if not encoder:
            print("Unable to create encoder")
            return False
        
        # Check if using software encoder (x264enc)
        encoder_name = encoder.get_factory().get_name()
        use_software_encoder = (encoder_name == "x264enc")
        
        # If using software encoder
        if use_software_encoder:
            print("Using software encoder, adding memory conversion")
            nvvidconv_cpu = Gst.ElementFactory.make("nvvideoconvert", "convertor-cpu")
            caps_cpu = Gst.ElementFactory.make("capsfilter", "caps-cpu")
            caps_cpu.set_property("caps", Gst.Caps.from_string("video/x-raw, format=I420"))
        else:
            nvvidconv_cpu = None
            caps_cpu = None
        
        h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
        if not h264parse:
            print("Unable to create h264parse")
            return False
        
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        rtppay.set_property("config-interval", 1)
        rtppay.set_property("pt", 96)
        rtppay.set_property("mtu", 1400)
        
        sink_rtsp = Gst.ElementFactory.make("udpsink", "udpsink")
        sink_rtsp.set_property("host", "127.0.0.1")
        sink_rtsp.set_property("port", self.config.udp_port_start)
        sink_rtsp.set_property("async", False)
        sink_rtsp.set_property("sync", False)
        sink_rtsp.set_property("max-lateness", -1)
        sink_rtsp.set_property("qos", False)
        
        # Kiểm tra các elements cần thiết
        required_elements = [streammux, pgie, nvvidconv, caps_rgba, tiler, nvosd,
                           queue_rtsp, nvvidconv_rtsp, caps_rtsp, encoder, h264parse, rtppay, sink_rtsp]
        
        if use_software_encoder:
            required_elements.extend([nvvidconv_cpu, caps_cpu])
        
        if not self.config.headless:
            required_elements.extend([tee, queue_display, nvvidconv_postosd, 
                                    caps_display, transform, sink_display])
        
        if not all(required_elements):
            print("Unable to create one or more elements")
            return False
        
        self.pipeline.add(streammux)
        self.pipeline.add(pgie)
        self.pipeline.add(nvvidconv)
        self.pipeline.add(caps_rgba)
        self.pipeline.add(tiler)
        self.pipeline.add(nvosd)
        
        # Thêm display elements nếu không headless
        if not self.config.headless:
            self.pipeline.add(tee)
            self.pipeline.add(queue_display)
            self.pipeline.add(nvvidconv_postosd)
            self.pipeline.add(caps_display)
            self.pipeline.add(transform)
            self.pipeline.add(sink_display)
        
        self.pipeline.add(queue_rtsp)
        self.pipeline.add(nvvidconv_rtsp)
        self.pipeline.add(caps_rtsp)
        if use_software_encoder:
            self.pipeline.add(nvvidconv_cpu)
            self.pipeline.add(caps_cpu)
        self.pipeline.add(encoder)
        self.pipeline.add(h264parse)
        self.pipeline.add(rtppay)
        self.pipeline.add(sink_rtsp)
        
        for i, stream_data in self.streams.items():
            source_bin = self.builder.create_source_bin(
                i, stream_data.uri, self.cb_newpad, self.decodebin_child_added)
            if not source_bin:
                return False
            
            self.pipeline.add(source_bin)
            
            srcpad = source_bin.get_static_pad("src")
            sinkpad = streammux.get_request_pad(f"sink_{i}")
            srcpad.link(sinkpad)
        
        streammux.link(pgie)
        pgie.link(nvvidconv)
        nvvidconv.link(caps_rgba)
        caps_rgba.link(tiler)
        tiler.link(nvosd)
        
        if not self.config.headless:
            nvosd.link(tee)
            
            # Display branch
            tee_display_pad = tee.get_request_pad("src_0")
            queue_display_pad = queue_display.get_static_pad("sink")
            tee_display_pad.link(queue_display_pad)
            
            queue_display.link(nvvidconv_postosd)
            nvvidconv_postosd.link(caps_display)
            caps_display.link(transform)
            transform.link(sink_display)
            
            # RTSP branch
            tee_rtsp_pad = tee.get_request_pad("src_1")
            queue_rtsp_pad = queue_rtsp.get_static_pad("sink")
            tee_rtsp_pad.link(queue_rtsp_pad)
        else:
            # Headless mode: nvosd -> queue_rtsp (trực tiếp)
            nvosd.link(queue_rtsp)
        
        # RTSP encoding pipeline
        queue_rtsp.link(nvvidconv_rtsp)
        nvvidconv_rtsp.link(caps_rtsp)
        
        if use_software_encoder:
            # NVMM -> CPU memory
            caps_rtsp.link(nvvidconv_cpu)
            nvvidconv_cpu.link(caps_cpu)
            caps_cpu.link(encoder)
        else:
            caps_rtsp.link(encoder)
        
        encoder.link(h264parse)
        h264parse.link(rtppay)
        rtppay.link(sink_rtsp)
        
        if pyds:
            pgie_src_pad = pgie.get_static_pad("src")
            if pgie_src_pad:
                pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, 
                                      self.probe_handler.pgie_src_pad_buffer_probe, 0)
            
            tiler_src_pad = tiler.get_static_pad("src")
            if tiler_src_pad:
                tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, 
                                       lambda pad, info, u_data: self.probe_handler.tiler_src_pad_buffer_probe(
                                           pad, info, u_data, 
                                           self.config.tiler_width, self.config.tiler_height,
                                           self.config.streammux_width, self.config.streammux_height), 0)
        
        # Debug probe cho encoder
        def encoder_src_probe(pad, info, u_data):
            return Gst.PadProbeReturn.OK
        
        encoder_src_pad = encoder.get_static_pad("src")
        if encoder_src_pad:
            encoder_src_pad.add_probe(Gst.PadProbeType.BUFFER, encoder_src_probe, 0)
        
        return True
    
    def cb_newpad(self, decodebin, decoder_src_pad, data):
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
        if name.find("decodebin") != -1:
            Object.connect("child-added", self.decodebin_child_added, user_data)
    
    def bus_call(self, bus, message, loop):
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
    
    def start(self):
        server = GstRtspServer.RTSPServer.new()
        server.props.service = "%d" % self.config.rtsp_port
        server.attach(None)
        
        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(
            '( udpsrc name=pay0 port=%d buffer-size=1048576 do-timestamp=true '
            'caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, '
            'encoding-name=(string)H264, payload=(int)96" )'
            % self.config.udp_port_start
        )
        factory.set_shared(True)
        factory.set_latency(0)  # Minimize latency
        server.get_mount_points().add_factory("/multi-camera", factory)
        
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        print(f"cameras: {len(self.streams)}")
        print(f"RTSP URL: rtsp://localhost:{self.config.rtsp_port}/multi-camera")
        if self.config.save_frames:
            print(f"Saving frames to: {self.config.output_dir}")
        print("Press Ctrl+C to stop\n")
        
        if self.frame_saver:
            self.frame_saver.start()
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            return
        
        state_change = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        
        self.stop()
    
    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
        
        if self.frame_saver:
            self.frame_saver.stop()
        
        if self.rtmp_stream_manager:
            self.rtmp_stream_manager.stop_all()
        
        print("Done")
