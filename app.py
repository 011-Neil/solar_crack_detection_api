# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
import torch
from PIL import Image
from io import BytesIO
import cv2
import base64
import time
import os

# ----------------------------
# Setup YOLOv5 custom model
# ----------------------------
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='exp/weights/best.pt',
    force_reload=True
)
model.eval()

CAMERA_SOURCES = {
    "pc": 0,        # default laptop/desktop webcam
    "droid": 1   # DroidCam (virtual webcam)
    # or use RTSP/HTTP if DroidCam is streaming over WiFi:
    # "droid": "http://192.168.x.x:4747/video"
}

if torch.cuda.is_available():
    model.to('cuda')
    torch.backends.cudnn.benchmark = True  # speed up for fixed-size input

# ----------------------------
# Flask app + SocketIO
# ----------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

latest_crack_count = 0
STREAMS = {}  # sid -> { "running": bool, "camera_id": any }

# ----------------------------
# IMAGE detection (unchanged)
@app.route('/api/detect-image', methods=['POST'])
def detect_image():
    try:
        print("started !!")

        # Make sure key matches frontend (use "image")
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        img_bytes = file.read()

        # Open image
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Run YOLO inference
        with torch.no_grad():
            results = model(img, size=640)

        # Confidence threshold
        conf_threshold = 0.6
        data = results.pandas().xyxy[0]

        cracks = []
        detected = False
        if hasattr(data, "iterrows"):
            for _, row in data.iterrows():
                if float(row["confidence"]) >= conf_threshold:
                    detected = True
                    cracks.append({
                        "class": row["name"],
                        "confidence": round(float(row["confidence"]), 3),
                        "box": [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
                    })

        # Render annotated image
        results.render()
        if hasattr(results, "ims"):
            img_array = results.ims[0]
        else:
            img_array = results.imgs[0]

        annotated_img = Image.fromarray(img_array)
        buf = BytesIO()
        annotated_img.save(buf, format="JPEG")
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "crack_detected": detected,
            "details": cracks,
            "annotated_image": img_base64
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

# ----------------------------
# VIDEO file detection (unchanged)
# ----------------------------
@app.route('/api/detect-video', methods=['POST'])
def detect_video():
    file = request.files['file']
    input_path = 'temp_input.mp4'
    output_path = 'temp_output.mp4'
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = max(1, int(cap.get(cv2.CAP_PROP_FPS)) or 25)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    details = []
    frame_count = 0
    conf_threshold = 0.6

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        with torch.no_grad():
            results = model(frame, size=640 if not torch.cuda.is_available() else 640)

        results.render()
        annotated_frame = results.ims[0] if hasattr(results, 'ims') else results.imgs[0]
        out.write(annotated_frame)

        data = results.pandas().xyxy[0]
        frame_cracks = []
        if hasattr(data, 'iterrows'):
            for _, row in data.iterrows():
                if float(row["confidence"]) >= conf_threshold:
                    frame_cracks.append({
                        "class": row["name"],
                        "confidence": float(row["confidence"]),
                        "box": [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
                    })
        details.append({"frame": frame_count, "cracks": frame_cracks})

    cap.release()
    out.release()

    with open(output_path, "rb") as vf:
        video_base64 = base64.b64encode(vf.read()).decode('utf-8')

    return jsonify({
        "crack_detected": any(len(f["cracks"]) for f in details),
        "details": details,
        "annotated_video": video_base64
    })

# ----------------------------
# WebSocket: ultra-smooth live stream
# ----------------------------
def open_capture(camera_id):
    """
    camera_id can be:
    - int index (0,1,2,...) for PC webcam / DroidCam (USB virtual cam)
    - a string URL (e.g., 'http://.../video', 'rtsp://...')
    """
    if isinstance(camera_id, str) and (camera_id.startswith('rtsp://') or camera_id.startswith('http')):
        return cv2.VideoCapture(camera_id)
    # default: treat as index
    try:
        cam_index = int(camera_id)
        
    except Exception:
        cam_index = 0
    return cv2.VideoCapture(cam_index)

def stream_loop(sid, camera_id):
    global latest_crack_count

    cap = open_capture(camera_id)
    conf_threshold = 0.6

    # Prefer slightly smaller inference size for speed; tweak if needed
    infer_size = 480

    # Try to reduce internal buffer (helps reduce latency)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while STREAMS.get(sid, {}).get("running", False):
        success, frame = cap.read()
        if not success or frame is None:
            socketio.sleep(0.005)
            continue

        frame = frame.astype('uint8')

        # YOLO inference
        with torch.no_grad():
            if torch.cuda.is_available():
                results = model(frame, size=infer_size)
            else:
                results = model(frame, size=infer_size)

        # Count cracks >= threshold
        data = results.pandas().xyxy[0]
        latest_crack_count = 0
        if hasattr(data, "iterrows"):
            for _, row in data.iterrows():
                if float(row["confidence"]) >= conf_threshold:
                    latest_crack_count += 1

        # Render bounding boxes on frame
        annotated_frame = frame.copy()
        data = results.pandas().xyxy[0]
        latest_crack_count = 0
        
        if hasattr(data, "iterrows"):
            for _, row in data.iterrows():
                conf = float(row["confidence"])
                if conf >= conf_threshold:
                    latest_crack_count += 1
                    x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                    cls_name = row["name"]
        
                    # Draw box + label
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        annotated_frame,
                        f"{cls_name} {conf:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
        

        # Overlay crack count
        cv2.putText(
            annotated_frame,
            f"Cracks: {latest_crack_count}",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # Encode + emit over socket
        ok, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            socketio.sleep(0.001)
            continue
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

        socketio.emit("video_frame", {"frame": frame_b64, "count": latest_crack_count}, to=sid)

        # Yield to event loop (controls frame-rate / reduces lag)
        socketio.sleep(0.001)

    cap.release()

@socketio.on("start_stream")
def handle_start_stream(data):
    sid = request.sid
    camera_type = data.get("camera_type", "pc")  # "pc" or "droid"
    camera_id = CAMERA_SOURCES.get(camera_type, 0)

    if STREAMS.get(sid, {}).get("running"):
        STREAMS[sid]["running"] = False

    STREAMS[sid] = {"running": True, "camera_id": camera_id}
    socketio.start_background_task(stream_loop, sid, camera_id)


@socketio.on("stop_stream")
def handle_stop_stream():
    sid = request.sid
    if STREAMS.get(sid, {}).get("running"):
        STREAMS[sid]["running"] = False

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    if STREAMS.get(sid, {}).get("running"):
        STREAMS[sid]["running"] = False

# (Optional) REST for count if you still want it
@app.route('/api/video-crack-count')
def get_live_crack_count():
    global latest_crack_count
    return jsonify({"crack_count": latest_crack_count})

if __name__ == "__main__":
    # Use eventlet or gevent for best SocketIO performance
    # pip install eventlet
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
