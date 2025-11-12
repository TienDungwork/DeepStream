
# DeepStream Video Analytics

## Yêu cầu hệ thống

- NVIDIA GPU với CUDA support
- DeepStream SDK 7.1
- Python 3.10
- OpenCV
- PyDS (DeepStream Python bindings)

### Chạy ứng dụng:
```bash
python3 polygon_detection_app.py \
  -i /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 \
  -c config_infer_primary_yolo11.txt \
  --save-frames \
  --output-dir output_frames
```

### Các tham số:
- `-i`, `--input`: Đường dẫn video đầu vào (bắt buộc)
- `-c`, `--config`: File cấu hình YOLO inference (bắt buộc)
- `-p`, `--polygon`: File JSON lưu polygon (mặc định: polygon.json)
- `-o`, `--output`: File video output (MP4)
- `--save-frames`: Lưu frame có đối tượng phát hiện
- `--output-dir`: Thư mục lưu output frames (mặc định: output_frames)
- `--rtsp-port`: Port RTSP server (mặc định: 8554)
- `--udp-port`: Port UDP sink (mặc định: 5400)
- `--no-draw-bbox`: Không vẽ bounding box trên frame lưu
- `--skip-prompt`: Bỏ qua prompt vẽ lại polygon

### RTSP Stream:
```
rtsp://localhost:8554/polygon-detect
```

## Camera System(C++)


```bash
deepstream-app -c deepstream_app_config.txt
```

## Yêu cầu hệ thống

- NVIDIA GPU với CUDA support
- DeepStream SDK 7.1
- Python 3.10
- OpenCV
- PyDS (DeepStream Python bindings)