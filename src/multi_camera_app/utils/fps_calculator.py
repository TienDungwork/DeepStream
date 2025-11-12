import time


class FPSCalculator:
    def __init__(self, update_interval: int = 30):
        self.update_interval = update_interval
        self.frame_count = 0
        self.start_time = None
        self.current_fps = 0.0
    
    def update(self) -> float:
        if self.start_time is None:
            self.start_time = time.time()
            self.frame_count = 0
        
        self.frame_count += 1
        
        if self.frame_count >= self.update_interval:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.current_fps = self.frame_count / elapsed_time
            self.start_time = time.time()
            self.frame_count = 0
        
        return self.current_fps
    
    def get_fps(self) -> float:
        return self.current_fps
    
    def reset(self):
        self.frame_count = 0
        self.start_time = None
        self.current_fps = 0.0
