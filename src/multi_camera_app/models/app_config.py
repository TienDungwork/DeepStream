from dataclasses import dataclass
import yaml
import os


@dataclass
class AppConfig:
    streams_config_file: str
    inference_config_file: str
    rtsp_port: int = 8554
    udp_port_start: int = 5400
    save_frames: bool = False
    output_dir: str = "output_frames"
    draw_bbox: bool = True
    skip_prompt: bool = False
    headless: bool = False
    
    streammux_width: int = 1920
    streammux_height: int = 1080
    streammux_batch_size: int = 4
    streammux_batched_push_timeout: int = 40000
    
    tiler_width: int = 1920
    tiler_height: int = 1080
    
    enable_perf_measurement: bool = True
    perf_measurement_interval: int = 5
    
    # RTMP Streaming settings
    enable_rtmp_streaming: bool = False
    rtmp_server_url: str = "rtmp://192.168.1.144:1935/POC"
    rtmp_stream_width: int = 1280
    rtmp_stream_height: int = 720
    rtmp_fps: int = 15
    rtmp_bitrate: int = 800000
    
    @classmethod
    def from_yaml(cls, config_path: str = None):
        if config_path is None:
            # Default config path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(
                os.path.dirname(current_dir),
                'resources',
                'configs',
                'app_config.yaml'
            )
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
