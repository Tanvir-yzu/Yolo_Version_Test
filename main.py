from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

app = Flask(__name__)

# Initialize YOLO model
model = YOLO("yolov8m.pt")

# Define class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ] # Your extended class names list here

def gen_frames():  # Generate frame by frame from camera
    cap = cv2.VideoCapture(0)
    prev_frame_time = 0
    new_frame_time = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            new_frame_time = time.time()

            results = model(frame, stream=True, device="mps")
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Bounding Box Color
                    color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
                    cvzone.cornerRect(frame, (x1, y1, w, h), colorR=color)
                    # Class Name
                    cls = int(box.cls[0])
                    cvzone.putTextRect(
                        frame,
                        f"{classNames[cls]} {conf}",
                        (max(0, x1), max(35, y1)),
                        scale=1,
                        thickness=1,
                    )

            # Calculate and display FPS
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cvzone.putTextRect(frame, f"FPS: {int(fps)}", (10, 50), scale=2, thickness=2)

            # Display the timestamp
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(new_frame_time)
            )
            cvzone.putTextRect(frame, timestamp, (10, 100), scale=2, thickness=2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
