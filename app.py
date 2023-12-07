from flask import Flask, Response, render_template, request, jsonify
import cv2
import torch
from yolov5 import YOLOv5
import requests
app = Flask(__name__)

# Load the YOLOv5 model
model = YOLOv5("yolov5s.pt", device="cpu")  # Specify the correct path to the .pt file

# Define the class names manually (for COCO dataset)
class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush","mouse"]  # Add or remove according to your model

def generate_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Adjust based on your system
    
    with torch.no_grad():  # Inference without gradient calculation
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Convert frame to the correct format for YOLOv5
                results = model.predict(frame)

                # Draw bounding boxes and labels on the frame
                for det in results.xyxy[0]:
                    # Extract xyxy bounding box, confidence, and class
                    xmin, ymin, xmax, ymax, conf, cls = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4], int(det[5])
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_names[cls]} {conf:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Video streaming home page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
