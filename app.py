from flask import Flask, Response, render_template, jsonify
import cv2
import os
import time
import threading
import numpy as np
from ultralytics import YOLO
from collections import deque

app = Flask(__name__)

MODEL_PATH = 'models/best.pt'
SAMPLE_VIDEOS_DIR = 'videotest'

# Các tham số quan trọng để tinh chỉnh phát hiện lửa
CONFIDENCE_THRESHOLD = 0.45  # Giảm ngưỡng để phát hiện chính xác hơn
DETECTION_PERSISTENCE = 3  # Số frame liên tiếp cần phát hiện lửa để xác nhận
TIME_WINDOW = 10  # Cửa sổ thời gian (số giây) để tính trung bình phát hiện
MIN_DETECTION_AREA = 500  # Diện tích tối thiểu (pixel^2) của bounding box để loại bỏ nhiễu

# Khởi tạo model
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Danh sách camera
CAMERAS = [
    {"id": 0, "name": "Raspberry Pi 1 (Webcam)", "type": "webcam", "source": 0},
    {"id": 1, "name": "Raspberry Pi 2 (Sample Video 1)", "type": "video", "source": "videotest/fire_sample.mp4"}
]

camera_streams = {}
frames = {}
fire_detected = {}
last_detection_time = {}
detection_history = {}  # Lưu lịch sử phát hiện của mỗi camera
fire_confidence = {}    # Lưu mức độ tin cậy mới nhất của phát hiện

def create_blank_frame(text):
    frame = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame

def init_cameras():
    for camera in CAMERAS:
        try:
            if camera["type"] == "webcam":
                stream = cv2.VideoCapture(camera["source"])
                if not stream.isOpened():
                    print(f"Failed to open webcam for {camera['name']}. Trying alternative source...")
                    stream = cv2.VideoCapture(1)
                    if not stream.isOpened():
                        frames[camera["id"]] = create_blank_frame(f"Cannot open webcam for {camera['name']}")
                        fire_detected[camera["id"]] = False
                        continue
            else:
                if not os.path.exists(camera["source"]):
                    frames[camera["id"]] = create_blank_frame(f"Video {camera['source']} not found")
                    fire_detected[camera["id"]] = False
                    continue
                stream = cv2.VideoCapture(camera["source"])
                if not stream.isOpened():
                    frames[camera["id"]] = create_blank_frame(f"Cannot open video for {camera['name']}")
                    fire_detected[camera["id"]] = False
                    continue

            camera_streams[camera["id"]] = stream
            frames[camera["id"]] = None
            fire_detected[camera["id"]] = False
            last_detection_time[camera["id"]] = 0
            detection_history[camera["id"]] = deque(maxlen=DETECTION_PERSISTENCE)
            fire_confidence[camera["id"]] = 0.0

            thread = threading.Thread(target=process_camera_frames, args=(camera["id"],))
            thread.daemon = True
            thread.start()
            print(f"Initialized camera: {camera['name']}")

        except Exception as e:
            print(f"Error initializing camera {camera['name']}: {e}")
            frames[camera["id"]] = create_blank_frame(f"Error: {str(e)}")
            fire_detected[camera["id"]] = False

def process_camera_frames(camera_id):
    global frames, fire_detected, last_detection_time, fire_confidence

    camera_info = next((c for c in CAMERAS if c["id"] == camera_id), None)
    stream = camera_streams.get(camera_id)

    if not stream or not stream.isOpened():
        frames[camera_id] = create_blank_frame(f"Cannot open {camera_info['name']}")
        return

    # Khởi tạo lịch sử phát hiện
    recent_detections = detection_history[camera_id]
    
    # Tạo danh sách theo dõi phát hiện theo thời gian (cho cửa sổ trượt)
    detection_timestamps = deque(maxlen=int(30 * TIME_WINDOW))  # Giả sử 30fps
    
    while True:
        success, frame = stream.read()

        if not success:
            if camera_info["type"] == "video":
                stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                frames[camera_id] = create_blank_frame(f"Lost connection to {camera_info['name']}")
                time.sleep(1)
                continue

        annotated_frame = frame.copy()
        
        # Thêm thông tin về camera vào góc trên bên trái với màu dễ nhìn
        cv2.putText(annotated_frame, camera_info["name"], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        current_time = time.time()
        # Giảm tần suất phát hiện để tiết kiệm CPU nhưng vẫn đảm bảo độ nhạy
        if current_time - last_detection_time.get(camera_id, 0) > 0.2:  # 5 lần mỗi giây
            try:
                if model:
                    # Phát hiện lửa với model
                    results = model(frame)
                    detections = results[0]
                    
                    # Biến để theo dõi phát hiện lửa trong frame hiện tại
                    current_frame_has_fire = False
                    best_confidence = 0.0
                    best_box = None
                    
                    for detection in detections.boxes.data.tolist():
                        x1, y1, x2, y2, confidence, class_id = detection
                        
                        # Chỉ xử lý lửa (class_id = 0) và vượt qua ngưỡng tin cậy
                        if confidence > CONFIDENCE_THRESHOLD and int(class_id) == 0:
                            # Tính diện tích bounding box
                            box_area = (x2 - x1) * (y2 - y1)
                            
                            # Lọc các phát hiện quá nhỏ (có thể là nhiễu)
                            if box_area >= MIN_DETECTION_AREA:
                                current_frame_has_fire = True
                                
                                # Lưu box tốt nhất để hiển thị
                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    best_box = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Cập nhật lịch sử phát hiện cho frame hiện tại
                    recent_detections.append(current_frame_has_fire)
                    detection_timestamps.append((current_time, current_frame_has_fire))
                    
                    # Chỉ vẽ bounding box cho phát hiện tốt nhất để tránh quá nhiều box chồng chéo
                    if best_box:
                        x1, y1, x2, y2 = best_box
                        fire_confidence[camera_id] = best_confidence
                        
                        # Vẽ bounding box với độ dày phù hợp
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # Thêm nhãn với mức độ tin cậy, được làm tròn đến 2 chữ số thập phân
                        confidence_text = f"Fire: {best_confidence:.2f}"
                        text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Vẽ nền cho văn bản để dễ đọc
                        cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10), 
                                     (x1 + text_size[0], y1), (0, 0, 0), -1)
                        
                        # Thêm văn bản với mức độ tin cậy
                        cv2.putText(annotated_frame, confidence_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Tính toán phát hiện lửa dựa trên số frame liên tiếp có phát hiện
                    fire_detected[camera_id] = sum(recent_detections) >= DETECTION_PERSISTENCE - 1
                    
                    # Nếu không có đủ lịch sử, đánh giá theo tỷ lệ
                    if len(recent_detections) < DETECTION_PERSISTENCE:
                        if len(recent_detections) > 0:
                            fire_detected[camera_id] = sum(recent_detections) / len(recent_detections) >= 0.5
                    
                    # Phân tích phát hiện trong cửa sổ thời gian
                    recent_period = [d for t, d in detection_timestamps if current_time - t <= TIME_WINDOW]
                    if recent_period:
                        detection_rate = sum(recent_period) / len(recent_period)
                        # Nếu tỷ lệ phát hiện trong cửa sổ thời gian cao, đánh dấu là phát hiện được lửa
                        if detection_rate >= 0.4:  # 40% frames có phát hiện
                            fire_detected[camera_id] = True
                
                last_detection_time[camera_id] = current_time
                
            except Exception as e:
                print(f"Error processing frame from {camera_info['name']}: {e}")

        # Hiển thị cảnh báo nếu phát hiện lửa
        if fire_detected.get(camera_id, False):
            # Vẽ cảnh báo lớn trên màn hình
            cv2.rectangle(annotated_frame, (40, 40), (600, 100), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "FIRE DETECTED!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Thêm thông tin về mức độ tin cậy
            confidence_info = f"Confidence: {fire_confidence.get(camera_id, 0):.2f}"
            cv2.putText(annotated_frame, confidence_info, (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        frames[camera_id] = annotated_frame
        time.sleep(0.03)  # Giảm tải CPU

def generate_frames(camera_id):
    while True:
        if camera_id not in frames or frames[camera_id] is None:
            time.sleep(0.1)
            continue
        try:
            ret, buffer = cv2.imencode('.jpg', frames[camera_id])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error generating frame for camera {camera_id}: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    if camera_id not in frames:
        return "Camera not found", 404
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    any_fire = any(fire_detected.values())
    camera_status = [{
        "id": cam["id"],
        "name": cam["name"],
        "fire_detected": fire_detected.get(cam["id"], False),
        "confidence": fire_confidence.get(cam["id"], 0.0) if fire_detected.get(cam["id"], False) else 0.0
    } for cam in CAMERAS]
    return jsonify({
        "any_fire_detected": any_fire,
        "cameras": camera_status
    })

@app.route('/check_cameras')
def check_cameras():
    info = []
    for cam in CAMERAS:
        cam_id = cam["id"]
        info.append({
            "id": cam_id,
            "name": cam["name"],
            "has_frame": cam_id in frames and frames[cam_id] is not None,
            "stream_opened": cam_id in camera_streams and camera_streams[cam_id].isOpened(),
            "detection_history": list(detection_history.get(cam_id, [])),
            "confidence": fire_confidence.get(cam_id, 0.0)
        })
    return jsonify(info)

if __name__ == '__main__':
    os.makedirs(SAMPLE_VIDEOS_DIR, exist_ok=True)
    init_cameras()
    app.run(host='0.0.0.0', port=5000, debug=True)