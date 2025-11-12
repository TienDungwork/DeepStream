import json
import yaml
import os
from typing import Dict
from multi_camera_app.models import StreamData, AppConfig
from multi_camera_app.controllers.polygon_controller import PolygonController
from multi_camera_app.controllers.pipeline_controller import PipelineController


class AppController:
    def __init__(self):
        self.config = None
        self.streams: Dict[int, StreamData] = {}
        self.polygon_controller = PolygonController()
        self.pipeline_controller = None
    
    def run(self):
        """Main entry point - load config and run application"""
        # Load configuration
        self.config = AppConfig.from_yaml()
        
        # Validate paths
        if not os.path.exists(self.config.streams_config_file):
            print(f"Error: Streams config not found: {self.config.streams_config_file}")
            return 1
        
        if not os.path.exists(self.config.inference_config_file):
            print(f"Error: Inference config not found: {self.config.inference_config_file}")
            return 1
        
        # Load streams
        self.load_streams()
        
        # Setup polygons
        if not self.setup_polygons():
            return 1
        
        # Create pipeline
        if not self.create_pipeline():
            print("Failed to create pipeline")
            return 1
        
        # Start
        self.start()
        return 0
    
    def load_streams(self):
        file_ext = os.path.splitext(self.config.streams_config_file)[1].lower()
        
        if file_ext == '.yaml' or file_ext == '.yml':
            with open(self.config.streams_config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                streams_list = config_data.get('cameras', [])
        else:  # JSON
            with open(self.config.streams_config_file, 'r') as f:
                config_data = json.load(f)
                streams_list = config_data.get('streams', [])
            
        for i, stream_config in enumerate(streams_list):
            stream_data = StreamData(
                stream_id=i,
                name=stream_config.get('name', f'Camera {i+1}'),
                uri=stream_config['uri'],
                polygon_file=stream_config.get('polygon_file', f'polygon_cam{i+1}.json')
            )
            
            points, ready, orig_width, orig_height = self.polygon_controller.load_polygon(stream_data.polygon_file)
            if ready:
                stream_data.polygon_points = points
                stream_data.polygon_ready = True
                stream_data.polygon_original_width = orig_width
                stream_data.polygon_original_height = orig_height
                #print(f"  {stream_data.name}: polygon {len(points)} points @ {orig_width}x{orig_height}")
            
            self.streams[i] = stream_data
        
        print(f"Loaded {len(self.streams)} cameras")
    
    def setup_polygons(self):
        for stream_id, stream_data in self.streams.items():
            if not stream_data.polygon_ready:
                points, success = self.polygon_controller.draw_polygon_ui(
                    stream_id, stream_data.name, stream_data.uri, stream_data.polygon_points)
                
                if not success:
                    return False
                
                stream_data.polygon_points = points
                stream_data.polygon_ready = True
                self.polygon_controller.save_polygon(stream_data.polygon_file, points)
            
            elif not self.config.skip_prompt:
                response = input(f"Redraw polygon for {stream_data.name}? (y/n): ").strip().lower()
                if response == 'y':
                    points, success = self.polygon_controller.draw_polygon_ui(
                        stream_id, stream_data.name, stream_data.uri, stream_data.polygon_points)
                    
                    if success:
                        stream_data.polygon_points = points
                        self.polygon_controller.save_polygon(stream_data.polygon_file, points)
        
        return True
    
    def create_pipeline(self):
        self.pipeline_controller = PipelineController(self.config, self.streams)
        return self.pipeline_controller.create_pipeline()
    
    def start(self):
        if self.pipeline_controller:
            self.pipeline_controller.start()
    
    def stop(self):
        if self.pipeline_controller:
            self.pipeline_controller.stop()
