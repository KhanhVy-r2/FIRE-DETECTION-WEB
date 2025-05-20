from flask import Flask, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # webcam máy thật

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Webcam is streaming. Visit /video_feed"

app.run(host='0.0.0.0', port=5000)