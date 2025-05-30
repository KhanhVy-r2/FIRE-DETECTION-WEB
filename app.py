from flask import Flask, Response, render_template, jsonify
from flask_mqtt import Mqtt
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import cv2
import time

app = Flask(__name__)
app.config['MQTT_BROKER_URL'] = 'broker.hivemq.com'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_TLS_ENABLED'] = False

mqtt = Mqtt(app)
devices = {}  # Lưu trạng thái và bounding box từ clients
LOG_DIR = 'logs'

# Cấu hình logging
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger('ServerLogger')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(os.path.join(LOG_DIR, 'server_log.txt'), maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
logger.addHandler(handler)

# Danh sách client (mô phỏng Raspberry Pi)
CAMERAS = [
    {"id": 0, "name": "Raspberry Pi 1", "source": 0},  # Webcam
    {"id": 1, "name": "Raspberry Pi 2", "source": "videotest/fire_sample.mp4"}  # Video mẫu
]

@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected to MQTT broker")
        mqtt.subscribe('fire/detection')
    else:
        logger.error("Failed to connect to MQTT broker")

@mqtt.on_message()
def handle_message(client, userdata, message):
    try:
        data = json.loads(message.payload.decode())
        devices[data['device_id']] = data
        logger.info(f"Received: {data}")
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS)

@app.route('/status')
def status():
    any_fire = any(device.get('status') == 'fire_detected' for device in devices.values())
    camera_status = [{
        "id": cam["id"],
        "name": cam["name"],
        "fire_detected": devices.get(cam["name"], {}).get('status') == 'fire_detected',
        "confidence": devices.get(cam["name"], {}).get('confidence', 0.0)
    } for cam in CAMERAS]
    return jsonify({
        "any_fire_detected": any_fire,
        "cameras": camera_status
    })

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    camera = next((c for c in CAMERAS if c["id"] == camera_id), None)
    if not camera:
        return "Camera not found", 404
    return Response(generate_frames(camera["source"], camera["name"]),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames(source, camera_name):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
    while True:
        success, frame = cap.read()
        if not success:
            error_frame = cv2.imread('static/error.jpg')
            if error_frame is None:
                error_frame = cv2.putText(
                    cv2.zeros((480, 640, 3), dtype=cv2.uint8),
                    "No video stream", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            ret, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        # Vẽ bounding box nếu có phát hiện cháy
        device_data = devices.get(camera_name, {})
        if device_data.get('status') == 'fire_detected' and 'boxes' in device_data:
            for box in device_data['boxes']:
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = box[4]
                # Vẽ khung đỏ
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Vẽ nhãn
                label = f"FIRE DETECTED! {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/check_cameras')
def check_cameras():
    info = []
    for cam in CAMERAS:
        cap = cv2.VideoCapture(cam["source"])
        has_frame = cap.isOpened()
        cap.release()
        info.append({
            "id": cam["id"],
            "name": cam["name"],
            "has_frame": has_frame,
            "confidence": devices.get(cam["name"], {}).get('confidence', 0.0)
        })
    return jsonify(info)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('videotest', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)