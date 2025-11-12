import argparse
import sys

sys.path.append("../")
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer
import math
import datetime
import time
from scipy.ndimage import median_filter
import numpy as np

import pyds


class EstimationSpeehMP4Out:
    def __init__(
        self,
        uri_name,
        fps,
        ll_lib_file,
        ll_config_file,
        config_file_path,
        path_output,
    ):
        self.uri_name = uri_name
        self.ll_lib_file = ll_lib_file
        self.ll_config_file = ll_config_file
        self.config_file_path = config_file_path
        self.number_sources = 1
        self.TILED_OUTPUT_WIDTH = 1920
        self.TILED_OUTPUT_HEIGHT = 1080
        self.fps = fps
        self.history_frame = []
        self.history_veloc = []
        self.path_output = path_output

        self.counting_car = 0
        self.counting_motobike = 0
        self.counting_bus = 0
        self.counting_truck = 0

        # All flag
        self.car_id = 2
        self.motobike_id = 3
        self.bus_id = 5
        self.truck_id = 7

        self.frame_num = 0
        self.history_veloc_show = []

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        # Draw
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                self.frame_num += 1
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list

            list_info = []
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Calculate midpoint coordinates
                left = obj_meta.rect_params.left
                top = obj_meta.rect_params.top
                width = obj_meta.rect_params.width
                height = obj_meta.rect_params.height
                x1 = left + width / 2
                y1 = top + height
                coordinates = (x1, y1)
                dict_obj = {
                    "uniqueID": obj_meta.object_id,
                    "coordinates": coordinates,
                    "velocity_kmh": None,
                }

                # Calculate velocity
                result = next(
                    (
                        item
                        for item in self.history_frame
                        if item["uniqueID"] == obj_meta.object_id
                    ),
                    None,
                )
                if result is not None:
                    coordinates = result["coordinates"]
                    x2 = coordinates[0]
                    y2 = coordinates[1]

                    rate = 0
                    if top >= 0 and top < 200:
                        rate = 4.750 / 23.116076206597757
                    if top >= 200 and top < 300:
                        rate = 4.750 / 47.9124824394614
                    if top >= 300 and top < 400:
                        rate = 4.750 / 98.13688303629557
                    if top > 400:
                        rate = 4.750 / 171.8491450718471

                    time = 1 / self.fps
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    velocity = (distance / time) * rate
                    velocity_kmh = velocity * 3.6
                    found_object_veloc = next(
                        (
                            obj
                            for obj in self.history_veloc
                            if obj["uniqueID"] == obj_meta.object_id
                        ),
                        None,
                    )
                    if found_object_veloc is None:
                        dict_voloc = {
                            "uniqueID": obj_meta.object_id,
                            "list_volec": [velocity_kmh],
                        }
                        self.history_veloc.append(dict_voloc)
                        obj_meta.text_params.display_text = (
                            f"{obj_meta.object_id} | {int(velocity_kmh)} km/h"
                        )
                        dict_obj["velocity_kmh"] = velocity_kmh

                        # Cập nhật lại list lịch sử hiện thị
                        object_volec_show = next(
                            (
                                obj
                                for obj in self.history_veloc_show
                                if obj["uniqueID"] == obj_meta.object_id
                            ),
                            None,
                        )
                        # Nếu mà rỗng thì khởi tạo đối tượng thêm vào list này
                        if object_volec_show is None:
                            new_object = {
                                "uniqueID": obj_meta.object_id,
                                "frameID": self.frame_num,
                                "veloc": velocity_kmh,
                            }
                            self.history_veloc_show.append(new_object)
                        else:
                            object_volec_show["frameID"] = self.frame_num
                            object_volec_show["veloc"] = velocity_kmh

                    else:
                        list_volec = found_object_veloc["list_volec"]
                        list_volec1 = np.append(list_volec, velocity_kmh)
                        found_object_veloc["list_volec"] = list_volec1

                        median_value = np.median(list_volec1[-70:])
                        if median_value < 4:
                            median_value = 0

                        dict_obj["velocity_kmh"] = median_value
                        v0 = result["velocity_kmh"]
                        new_veloc = median_value

                        # Logic xử lý hiện thị vận tốc
                        object_volec_show = next(
                            (
                                obj
                                for obj in self.history_veloc_show
                                if obj["uniqueID"] == obj_meta.object_id
                            ),
                            None,
                        )
                        if object_volec_show is None:
                            obj_meta.text_params.display_text = f"{obj_meta.obj_label} {obj_meta.object_id} | {int(new_veloc)} km/h"
                            new_object_history_veclo = {
                                "uniqueID": obj_meta.object_id,
                                "frameID": self.frame_num,
                                "veloc": velocity_kmh,
                            }
                            self.history_veloc_show.append(
                                new_object_history_veclo
                            )
                        else:
                            if (
                                self.frame_num - object_volec_show["frameID"]
                                > 60
                            ):
                                object_volec_show["frameID"] = self.frame_num
                                object_volec_show["veloc"] = new_veloc
                                v_show = new_veloc
                                if v_show < 7:
                                    v_show = 0
                                obj_meta.text_params.display_text = f"{obj_meta.obj_label} {obj_meta.object_id} | {int(v_show)} km/h"
                            else:
                                v_show = object_volec_show["veloc"]
                                if v_show < 7:
                                    v_show = 0
                                obj_meta.text_params.display_text = f"{obj_meta.obj_label} {obj_meta.object_id} | {int(v_show)} km/h"

                else:
                    obj_meta.text_params.display_text = (
                        f"{obj_meta.obj_label} | 0 km/h"
                    )
                    dict_obj["velocity_kmh"] = 0

                # Drawing box
                list_check = [
                    17,
                    37,
                    26,
                    125,
                    175,
                    227,
                    234,
                    327,
                    361,
                    296,
                    557,
                    602,
                ]
                if obj_meta.object_id not in list_check:
                    if obj_meta.class_id == 3:
                        obj_meta.rect_params.border_color.set(
                            0.0, 1.0, 0.0, 1.0
                        )
                    else:
                        obj_meta.rect_params.border_color.set(
                            0.0, 0.0, 1.0, 1.0
                        )
                else:
                    obj_meta.rect_params.border_color.set(0.0, 0.0, 0.0, 0.0)
                    obj_meta.text_params.display_text = ""

                # Couting
                l_user_meta = obj_meta.obj_user_meta_list
                while l_user_meta:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                        if (
                            user_meta.base_meta.meta_type
                            == pyds.nvds_get_user_meta_type(
                                "NVIDIA.DSANALYTICSOBJ.USER_META"
                            )
                        ):
                            user_meta_data = pyds.NvDsAnalyticsObjInfo.cast(
                                user_meta.user_meta_data
                            )
                            if user_meta_data.lcStatus:
                                if obj_meta.class_id == self.car_id:
                                    self.counting_car += 1
                                if obj_meta.class_id == self.motobike_id:
                                    self.counting_motobike += 1
                                if obj_meta.class_id == self.bus_id:
                                    self.counting_bus += 1
                                if obj_meta.class_id == self.truck_id:
                                    self.counting_truck += 1

                    except StopIteration:
                        break

                    try:
                        l_user_meta = l_user_meta.next
                    except StopIteration:
                        break

                # Draw Couting
                display_meta = pyds.nvds_acquire_display_meta_from_pool(
                    batch_meta
                )
                display_meta.num_labels = 1
                py_nvosd_text_params = display_meta.text_params[0]
                py_nvosd_text_params.display_text = (
                    "car:{} | motorcycle:{}".format(
                        self.counting_car, self.counting_motobike
                    )
                )
                py_nvosd_text_params.x_offset = self.TILED_OUTPUT_WIDTH - 370
                py_nvosd_text_params.y_offset = 0
                py_nvosd_text_params.font_params.font_name = "Serif"
                py_nvosd_text_params.font_params.font_size = 20
                py_nvosd_text_params.font_params.font_color.set(
                    1.0, 1.0, 1.0, 1.0
                )
                py_nvosd_text_params.set_bg_clr = 1
                py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

                list_info.append(dict_obj)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            self.history_frame = list_info
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def cb_newpad(self, decodebin, decoder_src_pad, data):
        print("In cb_newpad\n")
        caps = decoder_src_pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)
        print("gstname=", gstname)
        if gstname.find("video") != -1:
            print("features=", features)
            if features.contains("memory:NVMM"):
                bin_ghost_pad = source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    sys.stderr.write(
                        "Failed to link decoder src pad to source bin ghost pad\n"
                    )
            else:
                sys.stderr.write(
                    " Error: Decodebin did not pick nvidia decoder plugin.\n"
                )

    def decodebin_child_added(self, child_proxy, Object, name, user_data):
        print("Decodebin child added:", name, "\n")
        if name.find("decodebin") != -1:
            Object.connect("child-added", self.decodebin_child_added, user_data)

        pyds.configure_source_for_ntp_sync(hash(Object))

    def create_source_bin(self, index, uri):
        print("Creating source bin")
        bin_name = "source-bin-%02d" % index
        print(bin_name)
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            sys.stderr.write(" Unable to create source bin \n")

        uri_decode_bin = Gst.ElementFactory.make(
            "uridecodebin", "uri-decode-bin"
        )
        if not uri_decode_bin:
            sys.stderr.write(" Unable to create uri decode bin \n")
        # We set the input uri to the source element
        uri_decode_bin.set_property("uri", uri)

        uri_decode_bin.connect("pad-added", self.cb_newpad, nbin)
        uri_decode_bin.connect("child-added", self.decodebin_child_added, nbin)

        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(
            Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        )
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
        return nbin

    def main(self):
        Gst.init(None)
        pipeline = Gst.Pipeline()
        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        pipeline.add(streammux)
        source_bin = self.create_source_bin(0, uri_name)
        pipeline.add(source_bin)
        padname = "sink_%u" % 0
        sinkpad = streammux.get_request_pad(padname)
        srcpad = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        nvdsanalytics = Gst.ElementFactory.make(
            "nvdsanalytics", "nvds-analytics"
        )
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        nvvidconv_postosd = Gst.ElementFactory.make(
            "nvvideoconvert", "convertor_postosd"
        )
        caps = Gst.ElementFactory.make("capsfilter", "filter")
        encoder = Gst.ElementFactory.make("avenc_mpeg4", "encoder")
        encoder.set_property("bitrate", 5000000)
        codeparser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4-parser")
        container = Gst.ElementFactory.make("qtmux", "qtmux")
        sink = Gst.ElementFactory.make("filesink", "filesink")

        # Set up
        sink.set_property("location", self.path_output)

        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batch-size", self.number_sources)
        streammux.set_property("batched-push-timeout", 33000)
        streammux.set_property("attach-sys-ts", 0)
        pgie.set_property("config-file-path", self.config_file_path)

        tiler_rows = int(math.sqrt(self.number_sources))
        tiler_columns = int(math.ceil((1.0 * self.number_sources) / tiler_rows))
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_columns)
        tiler.set_property("width", self.TILED_OUTPUT_WIDTH)
        tiler.set_property("height", self.TILED_OUTPUT_HEIGHT)

        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)
        tracker.set_property("gpu-id", 0)
        tracker.set_property("ll-lib-file", self.ll_lib_file)
        tracker.set_property("ll-config-file", self.ll_config_file)

        nvdsanalytics.set_property("config-file", "nvdsanalytics_config.txt")

        # Connect element
        pipeline.add(pgie)
        pipeline.add(tracker)
        pipeline.add(nvdsanalytics)
        pipeline.add(tiler)
        pipeline.add(nvvidconv)
        pipeline.add(nvosd)
        pipeline.add(nvvidconv_postosd)
        pipeline.add(caps)
        pipeline.add(encoder)
        pipeline.add(codeparser)
        pipeline.add(container)
        pipeline.add(sink)

        streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(nvdsanalytics)
        nvdsanalytics.link(nvvidconv)
        nvvidconv.link(tiler)
        tiler.link(nvosd)
        nvosd.link(nvvidconv_postosd)
        nvvidconv_postosd.link(caps)
        caps.link(encoder)
        encoder.link(codeparser)
        codeparser.link(container)
        container.link(sink)

        # Create an event loop
        loop = GLib.MainLoop()
        osdsinkpad = nvosd.get_static_pad("sink")
        osdsinkpad.add_probe(
            Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0
        )

        # Start streaming
        print("Starting pipeline \n")
        pipeline.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        except BaseException:
            pass

        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    uri_name = "file:///opt/nvidia/deepstream/deepstream-6.3/src/video_test.mp4"
    ll_lib_file = "/opt/nvidia/deepstream/deepstream-6.3/lib/libnvds_nvmultiobjecttracker.so"
    ll_config_file = "config_tracker_NvDCF_perf.yml"
    config_file_path = "config_infer_primary_yolo11.txt"
    fps = 60
    path_output = "video_final_60.mp4"
    espeeh = EstimationSpeehMP4Out(
        uri_name=uri_name,
        fps=fps,
        ll_lib_file=ll_lib_file,
        ll_config_file=ll_config_file,
        config_file_path=config_file_path,
        path_output=path_output,
    )
    espeeh.main()
