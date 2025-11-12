import json
import os
from shapely.geometry import Point, Polygon
from typing import List, Tuple


class PolygonUtils:
    
    @staticmethod
    def load_polygon(polygon_file: str) -> Tuple[List[List[int]], bool, int, int]:
        if os.path.exists(polygon_file):
            try:
                with open(polygon_file, 'r') as f:
                    data = json.load(f)
                    points = data.get('points', [])
                    
                    # Load resolution
                    resolution = data.get('resolution', {'width': 1920, 'height': 1080})
                    width = resolution.get('width', 1920)
                    height = resolution.get('height', 1080)
                    
                    if len(points) >= 3:
                        return points, True, width, height
            except Exception as e:
                print(f"Error loading polygon: {e}")
        return [], False
    
    @staticmethod
    def save_polygon(polygon_file: str, points: List[List[int]], width: int = 1920, height: int = 1080):
        try:
            with open(polygon_file, 'w') as f:
                json.dump({
                    'points': points,
                    'resolution': {
                        'width': width,
                        'height': height
                    }
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving polygon: {e}")
    
    @staticmethod
    def is_point_in_polygon(x: float, y: float, polygon_points: List[List[int]]) -> bool:
        if len(polygon_points) < 3:
            return True
        
        try:
            point = Point(x, y)
            polygon = Polygon(polygon_points)
            return polygon.contains(point)
        except:
            return True
    
    @staticmethod
    def calculate_tile_position(stream_id: int, num_streams: int, 
                                tile_width: int, tile_height: int) -> Tuple[int, int, float, float]:
        import numpy as np
        
        tiler_rows = int(np.ceil(np.sqrt(num_streams)))
        tiler_cols = int(np.ceil(num_streams / tiler_rows))
        
        tile_row = stream_id // tiler_cols
        tile_col = stream_id % tiler_cols
        
        offset_x = tile_col * tile_width
        offset_y = tile_row * tile_height
        
        scale_x = tile_width / 1920.0
        scale_y = tile_height / 1080.0
        
        return offset_x, offset_y, scale_x, scale_y
