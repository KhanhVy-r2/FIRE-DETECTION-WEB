# FIRE-DETECTION-WEB

## Mô tả dự án
**FIRE-DETECTION-WEB** là hệ thống giả lập phát hiện cháy (lửa và khói) theo thời gian thực, được thiết kế để giám sát môi trường datacenter ở mức vật lý, sử dụng thiết bị mạng biên như Raspberry Pi. Hệ thống được xây dựng theo mô hình client-server, tích hợp mô hình YOLOv8 Nano (`best.pt`) để phát hiện cháy, gửi cảnh báo qua MQTT, và hiển thị luồng video với khung nhận diện lửa (bounding box) trên giao diện web.

### Tính năng chính
- **Phát hiện cháy thời gian thực**: Sử dụng YOLOv8 Nano để phát hiện lửa/khói từ webcam hoặc video mẫu, với độ chính xác mAP50 95.90%, mAP50-95 91.46%.
- **Hiển thị trực quan**: Giao diện web (`index.html`) hiển thị trạng thái hệ thống ("Bình thường" hoặc "CẢNH BÁO CHÁY!"), luồng video với khung đỏ và nhãn "FIRE DETECTED!", mức độ tin cậy (confidence).
- **Truyền thông MQTT**: Client gửi cảnh báo và tọa độ bounding box qua MQTT (`broker.hivemq.com`, topic: `fire/detection`).
- **Log chi tiết**: Ghi FPS, độ trễ, và cảnh báo vào `logs/fire_detection_log.txt` (client) và `logs/server_log.txt` (server).
- **Tối ưu cho Raspberry Pi**: Độ phân giải 320x320, giới hạn 5 FPS, và xử lý tại chỗ để giảm độ trễ.


## Yêu cầu hệ thống
- **Phần cứng**:
  - PC: Intel Core i5 hoặc tương đương, 8GB RAM, webcam USB.
  - Raspberry Pi (khuyến nghị): Raspberry Pi 4/5, 4GB RAM, Raspberry Pi Camera Module hoặc webcam USB.
- **Phần mềm**:
  - Hệ điều hành: Ubuntu 22.04/Windows 11 (PC) hoặc Raspberry Pi OS (Raspberry Pi).
  - Python: 3.10 trở lên.
  - Thư viện Python:
    ```
    flask==2.3.3
    flask-mqtt==1.2.1
    opencv-python-headless==4.8.0.76
    ultralytics==8.0.196
    paho-mqtt==1.6.1
    numpy==1.26.4
    ```

## Hướng dẫn cài đặt
1. **Clone repository**:
   ```bash
   git clone <repository_url>
   cd FIRE-DETECTION-WEB
   ```

2. **Tạo môi trường ảo (khuyến nghị)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Cài đặt thư viện**:
   ```bash
   pip install flask==2.3.3 flask-mqtt==1.2.1 opencv-python-headless==4.8.0.76 ultralytics==8.0.196 paho-mqtt==1.6.1 numpy==1.26.4
   ```

4. **Trên Raspberry Pi (tuỳ chọn)**:
   - Thay webcam trong `client1.py` bằng Raspberry Pi Camera Module hoặc RTSP stream:
     ```python
     cap = cv2.VideoCapture("rtsp://<pi_ip>:8554/motion")
     ```
   - Cài đặt Mosquitto để chạy MQTT cục bộ (giảm độ trễ):
     ```bash
     sudo apt install mosquitto
     sudo systemctl enable mosquitto
     ```
     Sửa `MQTT_BROKER = 'localhost'` trong `app.py`, `client1.py`, `client2.py`.

## Hướng dẫn sử dụng
1. **Khởi động server**:
   ```bash
   python app.py
   ```
   Server chạy tại `http://localhost:5000`.

2. **Khởi động client**:
   - Client 1 (webcam):
     ```bash
     python client1.py
     ```
   - Client 2 (video mẫu):
     ```bash
     python client2.py
     ```

3. **Truy cập giao diện web**:
   - Mở trình duyệt, truy cập `http://localhost:5000`.
   - Xem trạng thái hệ thống, luồng video với khung nhận diện, và mức độ tin cậy.
   - Sử dụng nút "Kiểm tra camera" để gỡ lỗi (`/check_cameras`).

4. **Kiểm tra log**:
   - Client: `logs/fire_detection_log.txt` (FPS, độ trễ, cảnh báo).
   - Server: `logs/server_log.txt` (dữ liệu MQTT).

## Thông số kỹ thuật
- **Mô hình**: YOLOv8 Nano (`best.pt`), mAP50 95.90%, mAP50-95 91.46%.
- **Thông số phát hiện**:
  - Ngưỡng tin cậy: 0.45 (`CONFIDENCE_THRESHOLD`).
  - Diện tích tối thiểu: 500px² (`MIN_DETECTION_AREA`).
  - Số frame liên tiếp: 3 (`DETECTION_PERSISTENCE`).
  - Cửa sổ thời gian: 10 giây (`TIME_WINDOW`), tỷ lệ phát hiện ≥ 40%.
- **Video**:
  - Độ phân giải: 320x320 (`imgsz=320`).
  - Tốc độ xử lý: Giới hạn 5 FPS (`if time.time() - last_detection_time > 0.2`).
- **MQTT**:
  - Broker: `broker.hivemq.com`, port 1883.
  - Topic: `fire/detection`.

## Hạn chế và cải tiến
1. **Hạn chế tài nguyên trên Raspberry Pi**:
   - **Vấn đề**: FPS thấp (~3-5), độ trễ cao (~50-150ms) do CPU/RAM hạn chế.
   - **Giải pháp**:
     - Xuất mô hình sang ONNX:
       ```bash
       yolo export model=models/best.pt format=onnx
       ```
     - Dùng `onnxruntime` để tăng tốc suy luận.
     - Giảm `imgsz=224`.

2. **Phụ thuộc mạng**:
   - **Vấn đề**: Mất kết nối MQTT (`broker.hivemq.com`) gây gián đoạn.
   - **Giải pháp**: Sử dụng Mosquitto cục bộ (xem phần cài đặt).

3. **Báo động giả**:
   - **Vấn đề**: Ánh sáng mạnh/vật đỏ có thể bị nhầm thành lửa.
   - **Giải pháp**:
     - Tăng ngưỡng tin cậy 0.6.
     - Thêm cảm biến nhiệt/CO2.

## Thông tin bổ sung
- **Đồ án**: Tìm hiểu phát hiện cháy trên thiết bị mạng biên để bảo vệ datacenter (mức vật lý).
- **Công cụ gỡ lỗi**:
  - Endpoint `/check_cameras`: Trả về trạng thái camera và confidence (JSON).
  - Log: Xem `logs/` để phân tích FPS, độ trễ, và cảnh báo.
- **Khả năng mở rộng**:
  - Thêm client: Tạo `client3.py` với `DEVICE_ID` mới, cập nhật `CAMERAS` trong `app.py`.
  - Tích hợp chữa cháy: Thêm endpoint trong `app.py` để gửi tín hiệu qua REST API hoặc GPIO relay.

## Liên hệ
Nếu bạn có thắc mắc hoặc cần hỗ trợ, vui lòng liên hệ qua email: [23521831@gm.uit.edu.vn], [23521621@gm.uit.edu.vn].
