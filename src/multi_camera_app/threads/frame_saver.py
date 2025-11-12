import threading
import queue
import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any


class FrameSaverThread:
    def __init__(self, output_dir: str, draw_bbox: bool = True, max_queue_size: int = 60):
        self.output_dir = output_dir
        self.draw_bbox = draw_bbox
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.thread = None
        self.stop_flag = False
    
    def start(self):
        if self.thread is None:
            self.stop_flag = False
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
    
    def stop(self):
        if self.thread is not None:
            self.stop_flag = True
            self.frame_queue.put(None)
            self.thread.join(timeout=5)
            self.thread = None
    
    def add_frame(self, stream_id: int, stream_name: str, frame_array: np.ndarray, 
                  objects: List[Dict], frame_number: int, polygon_points: List[List[int]]):
        try:
            frame_data = {
                'stream_id': stream_id,
                'stream_name': stream_name,
                'frame_array': frame_array.copy(),
                'objects': [obj.copy() for obj in objects],
                'frame_number': frame_number,
                'polygon_points': polygon_points
            }
            self.frame_queue.put(frame_data, block=False)
        except queue.Full:
            pass
    
    def _worker(self):
        while not self.stop_flag:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                
                if frame_data is None:
                    break
                
                self._save_frame(frame_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame saver thread: {e}")
    
    def _save_frame(self, frame_data: Dict[str, Any]):
        try:
            stream_id = frame_data['stream_id']
            stream_name = frame_data['stream_name']
            frame_array = frame_data['frame_array']
            objects = frame_data['objects']
            frame_number = frame_data['frame_number']
            polygon_points = frame_data['polygon_points']
            
            frame_to_save = frame_array.copy()
            
            if self.draw_bbox:
                if len(polygon_points) >= 3:
                    pts = np.array(polygon_points, np.int32)
                    cv2.polylines(frame_to_save, [pts], True, (0, 255, 0), 3)
                    overlay = frame_to_save.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.15, frame_to_save, 0.85, 0, frame_to_save)
                
                for obj in objects:
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
            
            cam_dir = os.path.join(self.output_dir, f"camera_{stream_id+1}")
            os.makedirs(cam_dir, exist_ok=True)
            
            frame_filename = os.path.join(cam_dir, f"frame_{frame_number:06d}.jpg")
            cv2.imwrite(frame_filename, frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            json_filename = os.path.join(cam_dir, f"frame_{frame_number:06d}.json")
            detection_data = {
                'stream_id': stream_id,
                'stream_name': stream_name,
                'frame_number': frame_number,
                'num_objects': len(objects),
                'polygon_points': polygon_points,
                'objects': objects
            }
            with open(json_filename, 'w') as f:
                json.dump(detection_data, f, indent=2)
            
        except Exception as e:
            print(f"Error saving frame: {e}")
