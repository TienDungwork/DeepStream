import threading
import subprocess
import numpy as np
import cv2
from queue import Queue, Empty
import time


class RTMPStreamer(threading.Thread):
    
    def __init__(self, camera_name: str, camera_id: int, stream_url: str,
                 frame_queue: Queue, stream_width: int = 1280, 
                 stream_height: int = 720, fps: int = 15, bitrate: int = 800000):
        super().__init__(daemon=True)
        self.camera_name = camera_name
        self.camera_id = camera_id
        self.frame_queue = frame_queue
        self.stream_width = stream_width
        self.stream_height = stream_height
        self.fps = fps
        self.bitrate = bitrate
        self.stream_endpoint = f"{stream_url}/camera_{camera_id}"
        
        self._stop_event = threading.Event()
        self._running = False
        self.encoder_process = None
        self.frames_sent = 0
        self.start_time = None
    
    def _init_encoder(self):
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-re', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', f'{self.stream_width}x{self.stream_height}',
                '-r', str(self.fps), '-i', 'pipe:0', '-pix_fmt', 'yuv420p',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
                '-b:v', str(self.bitrate), '-maxrate', str(self.bitrate),
                '-bufsize', str(self.bitrate // 2), '-g', str(self.fps * 2),
                '-keyint_min', str(self.fps), '-sc_threshold', '0',
                '-flush_packets', '1', '-f', 'flv', self.stream_endpoint,
                '-loglevel', 'error'
            ]
            
            self.encoder_process = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            print(f"[{self.camera_name}] Streaming to {self.stream_endpoint}")
            return True
        except Exception as e:
            print(f"[{self.camera_name}] Encoder init error: {e}")
            return False
    
    def start_streaming(self):
        self._init_encoder()
        self.start_time = time.time()
    
    def run(self):
        self._running = True
        self.start_streaming()
        
        while not self._stop_event.is_set() and self._running:
            try:
                item = self.frame_queue.get(timeout=1.0)
                
                if isinstance(item, tuple):
                    frame, metadata = item
                else:
                    frame = item
                    metadata = None
                
                while self.frame_queue.qsize() > 2:
                    try:
                        item = self.frame_queue.get_nowait()
                        if isinstance(item, tuple):
                            frame, metadata = item
                        else:
                            frame = item
                            metadata = None
                    except Empty:
                        break
                
                if frame.shape[:2] != (self.stream_height, self.stream_width):
                    frame = cv2.resize(frame, (self.stream_width, self.stream_height))
                
                if metadata:
                    self._draw_overlay(frame, metadata)
                
                try:
                    self.encoder_process.stdin.write(frame.tobytes())
                except Exception as e:
                    print(f"[{self.camera_name}] Write error: {e}")
                    self._reinit_encoder()
                
                self.frames_sent += 1
                
                if self.start_time and self.frames_sent % (self.fps * 5) == 0:
                    elapsed = time.time() - self.start_time
                    actual_fps = self.frames_sent / elapsed if elapsed > 0 else 0
                    print(f"[{self.camera_name}] Sent: {self.frames_sent}, FPS: {actual_fps:.1f}")
                
            except Empty:
                continue
            except Exception as e:
                print(f"[{self.camera_name}] Error: {e}")
                time.sleep(0.1)
        
        self.cleanup()
    
    def _draw_overlay(self, frame, metadata):
        try:
            streammux_width = metadata.get('streammux_width', 1920)
            streammux_height = metadata.get('streammux_height', 1080)
            
            scale_x = self.stream_width / streammux_width
            scale_y = self.stream_height / streammux_height
            
            polygon = metadata.get('polygon', [])
            if len(polygon) >= 3:
                scaled_polygon = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in polygon]
                pts = np.array(scaled_polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            objects = metadata.get('objects', [])
            for obj in objects:
                x, y, w, h = obj['bbox']
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{obj['class_name']}: {obj['confidence']:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            info_text = f"{metadata.get('camera_name', self.camera_name)} | FPS: {metadata.get('fps', 0):.2f} | Obj: {len(objects)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        except:
            pass
    
    def _reinit_encoder(self):
        try:
            if self.encoder_process:
                self.encoder_process.kill()
                self.encoder_process.wait(timeout=2)
            time.sleep(0.5)
            self._init_encoder()
        except:
            pass
    
    def stop(self):
        self._stop_event.set()
        self._running = False
    
    def cleanup(self):
        try:
            if self.encoder_process:
                self.encoder_process.stdin.close()
                self.encoder_process.kill()
                self.encoder_process.wait(timeout=2)
        except:
            pass


class RTMPStreamManager:
    
    def __init__(self, rtmp_server_url: str, stream_width: int = 1280,
                 stream_height: int = 720, fps: int = 15, bitrate: int = 800000):
        self.rtmp_server_url = rtmp_server_url
        self.stream_width = stream_width
        self.stream_height = stream_height
        self.fps = fps
        self.bitrate = bitrate
        self.streamers = {}
        self.frame_queues = {}
        
    def create_streamer(self, camera_id: int, camera_name: str):
        if camera_id in self.streamers:
            return
        
        frame_queue = Queue(maxsize=30)
        self.frame_queues[camera_id] = frame_queue
        
        streamer = RTMPStreamer(
            camera_name=camera_name,
            camera_id=camera_id,
            stream_url=self.rtmp_server_url,
            frame_queue=frame_queue,
            stream_width=self.stream_width,
            stream_height=self.stream_height,
            fps=self.fps,
            bitrate=self.bitrate
        )
        
        self.streamers[camera_id] = streamer
        streamer.start()
        
    def push_frame(self, camera_id: int, frame: np.ndarray, metadata: dict = None):
        if camera_id not in self.frame_queues:
            return
        
        try:
            if metadata:
                self.frame_queues[camera_id].put_nowait((frame, metadata))
            else:
                self.frame_queues[camera_id].put_nowait(frame)
        except:
            pass
    
    def stop_all(self):
        for camera_id, streamer in self.streamers.items():
            streamer.stop()
        for camera_id, streamer in self.streamers.items():
            streamer.join(timeout=5)
    
    def get_stream_url(self, camera_id: int) -> str:
        return f"{self.rtmp_server_url}/camera_{camera_id}"
