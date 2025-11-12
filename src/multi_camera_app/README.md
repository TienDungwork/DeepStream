
### Config
```yaml

# Có vẽ lại polygon không?
skip_prompt: true

# Chạy không hiển thị màn hình? (true = chỉ RTSP, false = hiển thị + RTSP)
headless: true

# Có lưu frames khi phát hiện đối tượng không?
save_frames: false
output_dir: "output_frames"        # Thư mục lưu frames
draw_bbox: true                    # Vẽ bbox lên ảnh lưu

# Độ phân giải xử lý
streammux_width: 1080
streammux_height: 720
# Port
rtsp_port: 8554
```

##  Camera
```yaml
# camera_config.yaml
cameras:
  - name: cam 1
    uri: file:///home/atin/Videos/camera.mp4
  
  - name: Cam 2
    uri: file:///home/atin/Videos/camera.mp4
```

### RTSP Stream
```
URL: rtsp://localhost:8554/multi-camera
```
- ffplay: `ffplay rtsp://localhost:8554/multi-camera`
