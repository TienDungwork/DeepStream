from dataclasses import dataclass, field
from typing import List, Tuple
import time


@dataclass
class StreamData:
    stream_id: int
    name: str
    uri: str
    polygon_file: str
    polygon_points: List[List[int]] = field(default_factory=list)
    polygon_ready: bool = False
    polygon_original_width: int = 1920   # Resolution where polygon was drawn
    polygon_original_height: int = 1080

    frame_count: int = 0
    fps: float = 0.0
    fps_start_time: float = None
    fps_frame_count: int = 0
    fps_update_interval: int = 30
    
    saved_frame_count: int = 0
    
    def scale_polygon_to_resolution(self, target_width: int, target_height: int) -> List[List[int]]:
        if not self.polygon_points:
            return []
        
        scale_x = target_width / self.polygon_original_width
        scale_y = target_height / self.polygon_original_height
        
        return [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in self.polygon_points]
    
    def update_fps(self):
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
    
    def increment_frame(self):
        self.frame_count += 1


@dataclass
class DetectionObject:
    object_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
