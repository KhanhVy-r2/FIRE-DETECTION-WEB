import cv2
import time
import paho.mqtt.client as mqtt
import json
from ultralytics import YOLO
from collections import deque
import logging
from logging.handlers import RotatingFileHandler
import os

# Cấu hình
MODEL_PATH = 'models/best.pt'
MQTT_BROKER = 'broker.hivemq.com'
MQTT_PORT = 1883
MQTT_TOPIC = 'fire/detection'
DEVICE_ID = 'Raspberry Pi 1'
CONFIDENCE_THRESHOLD = 0.45
DETECTION_PERSISTENCE = 3
TIME_WINDOW = 10
MIN_DETECTION_AREA = 500
LOG_DIR = 'logs'

# Cấu hình logging
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger('FireDetection')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(os.path.join(LOG_DIR, 'fire_detection_log.txt'), maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
logger.addHandler(handler)

# Load mô hình YOLOv8
try:
    model = YOLO(MODEL_PATH, task='detect')
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    exit()

# Kết nối MQTT
client = mqtt.Client()
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    logger.info(f"Connected to MQTT broker {MQTT_BROKER}")
except Exception as e:
    logger.error(f"Failed to connect to MQTT: {e}")
    exit()

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error(f"Cannot open webcam for {DEVICE_ID}")
    exit()

# Khởi tạo lịch sử phát hiện
recent_detections = deque(maxlen=DETECTION_PERSISTENCE)
detection_timestamps = deque(maxlen=int(30 * TIME_WINDOW))
last_detection_time = 0
fire_confidence = 0.0
frame_count = 0
start_time = time.time()
fps_log_interval = 10

try:
    while True:
        loop_start_time = time.time()
        success, frame = cap.read()
        if not success:
            logger.error(f"Failed to read frame for {DEVICE_ID}")
            time.sleep(1)
            continue

        # Xử lý YOLOv8 với tần suất 5 FPS
        if time.time() - last_detection_time > 0.2:
            results = model(frame, imgsz=320)
            current_frame_has_fire = False
            best_confidence = 0.0
            boxes = []

            for detection in results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = detection
                if confidence > CONFIDENCE_THRESHOLD and int(class_id) == 0:
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area >= MIN_DETECTION_AREA:
                        current_frame_has_fire = True
                        best_confidence = max(best_confidence, confidence)
                        boxes.append([x1, y1, x2, y2, confidence])

            recent_detections.append(current_frame_has_fire)
            detection_timestamps.append((time.time(), current_frame_has_fire))
            fire_detected = sum(recent_detections) >= DETECTION_PERSISTENCE - 1

            recent_period = [d for t, d in detection_timestamps if time.time() - t <= TIME_WINDOW]
            if recent_period and sum(recent_period) / len(recent_period) >= 0.4:
                fire_detected = True

            if fire_detected:
                fire_confidence = best_confidence
                alert_data = {
                    'device_id': DEVICE_ID,
                    'timestamp': time.time(),
                    'status': 'fire_detected',
                    'confidence': fire_confidence,
                    'boxes': boxes
                }
                client.publish(MQTT_TOPIC, json.dumps(alert_data))
                logger.info(f"FIRE DETECTED | Confidence: {fire_confidence:.2f} | Boxes: {len(boxes)}")

            last_detection_time = time.time()

        # Tính FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= fps_log_interval:
            fps = frame_count / elapsed_time
            logger.info(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

        # Tính độ trễ
        loop_time = (time.time() - loop_start_time) * 1000
        logger.info(f"Loop Latency: {loop_time:.2f}ms")

        time.sleep(0.03)

except KeyboardInterrupt:
    logger.info(f"Stopping {DEVICE_ID}")
finally:
    cap.release()
    client.disconnect()