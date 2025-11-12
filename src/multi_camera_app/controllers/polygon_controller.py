import cv2
import numpy as np
from typing import List, Dict
from multi_camera_app.utils import PolygonUtils


class PolygonController:
    def __init__(self):
        self.polygon_utils = PolygonUtils()
    
    def draw_polygon_ui(self, stream_id: int, stream_name: str, video_uri: str,
                       existing_points: List[List[int]] = None) -> tuple:
        print(f"POLYGON DRAWING MODE - {stream_name}")
        print("Left click: Add point | Right click: Remove | 'c': Clear | 's': Save | 'q': Quit")
        
        uri = video_uri.replace("file://", "")
        if uri.startswith("rtsp://"):
            print(f"RTSP stream detected. Using placeholder for drawing.")
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(frame, f"{stream_name} - Draw polygon area", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        else:
            cap = cv2.VideoCapture(uri)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"Error: Cannot read video from {uri}")
                return None, False
        
        original_frame = frame.copy()
        polygon_points = existing_points.copy() if existing_points else []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                polygon_points.append([x, y])
            elif event == cv2.EVENT_RBUTTONDOWN:
                if polygon_points:
                    polygon_points.pop()
        
        window_name = f'Draw Polygon - {stream_name}'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            display_frame = original_frame.copy()
            
            if len(polygon_points) > 0:
                for i, point in enumerate(polygon_points):
                    cv2.circle(display_frame, tuple(point), 5, (0, 255, 0), -1)
                    cv2.putText(display_frame, str(i+1), tuple(point), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                if len(polygon_points) > 1:
                    pts = np.array(polygon_points, np.int32)
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                    
                    if len(polygon_points) >= 3:
                        overlay = display_frame.copy()
                        cv2.fillPoly(overlay, [pts], (0, 255, 0))
                        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
            
            cv2.putText(display_frame, f"{stream_name} - Points: {len(polygon_points)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 's' to save and continue", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if len(polygon_points) >= 3:
                    cv2.destroyAllWindows()
                    return polygon_points, True
                else:
                    print("Need at least 3 points!")
            elif key == ord('c'):
                polygon_points = []
            elif key == 27:
                cv2.destroyAllWindows()
                return None, False
        
        return None, False
    
    def load_polygon(self, polygon_file: str) -> tuple:
        return self.polygon_utils.load_polygon(polygon_file)
    
    def save_polygon(self, polygon_file: str, points: List[List[int]]):
        self.polygon_utils.save_polygon(polygon_file, points)
