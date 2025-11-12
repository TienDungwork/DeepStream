import sys
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst


class GStreamerBuilder:
    
    @staticmethod
    def create_source_bin(index, uri, cb_newpad, decodebin_child_added):
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
        uri_decode_bin.connect("pad-added", cb_newpad, nbin)
        uri_decode_bin.connect("child-added", decodebin_child_added, nbin)
        
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write("Failed to add ghost pad in source bin\n")
            return None
        
        return nbin
    
    @staticmethod
    def create_streammux(width, height, batch_size, timeout):
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        if not streammux:
            return None
        
        streammux.set_property('width', width)
        streammux.set_property('height', height)
        streammux.set_property('batch-size', batch_size)
        streammux.set_property('batched-push-timeout', timeout)
        streammux.set_property('live-source', 1)
        streammux.set_property('nvbuf-memory-type', 0)
        streammux.set_property('gpu-id', 0)
        return streammux
    
    @staticmethod
    def create_pgie(config_file):
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            return None
        pgie.set_property('config-file-path', config_file)
        pgie.set_property('gpu-id', 0)
        return pgie
    
    @staticmethod
    def create_tiler(rows, cols, width, height):
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        if not tiler:
            return None
        
        tiler.set_property("rows", rows)
        tiler.set_property("columns", cols)
        tiler.set_property("width", width)
        tiler.set_property("height", height)
        return tiler
    
    @staticmethod
    def create_encoder():
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        if encoder:
            encoder.set_property("bitrate", 8000000)
            encoder.set_property("preset-level", 1)
            encoder.set_property("insert-sps-pps", 1)
            encoder.set_property("bufapi-version", 1)
            encoder.set_property("maxperf-enable", 1)
            return encoder
        
        # Try omxh264enc
        encoder = Gst.ElementFactory.make("omxh264enc", "encoder")
        if encoder:
            encoder.set_property("bitrate", 8000000)
            encoder.set_property("preset-level", 3)
            encoder.set_property("insert-sps-pps", True)
            encoder.set_property("control-rate", 2)
            return encoder
        
        # Fallback to x264enc
        encoder = Gst.ElementFactory.make("x264enc", "encoder")
        if encoder:
            encoder.set_property("bitrate", 8000)
            encoder.set_property("speed-preset", "ultrafast")
            encoder.set_property("tune", "zerolatency")
            encoder.set_property("key-int-max", 30)
        return encoder
