
# Real-time Object Detection Web App 🚀

This project implements a **real-time object detection web application** using the YOLOv8 model, Flask for web hosting, and OpenCV for video processing. It captures live camera feed, detects objects in the frame, and displays them with bounding boxes, class labels, confidence scores, and FPS (frames per second). 

## Features
- Live camera feed streaming 🎥
- Real-time object detection using **YOLOv8** model ⚡
- Displays bounding boxes, class labels, and confidence scores 📦
- FPS calculation for performance monitoring ⚙️
- Shows current timestamp on video stream 🕒

## Tech Stack 🛠️
- **Python**
- **Flask** for web application
- **OpenCV** for camera access and image processing
- **YOLOv8** model for object detection (via Ultralytics)
- **cvzone** for drawing bounding boxes and FPS display

## Demo 🖥️
![Real-time Detection Demo](demo_image.gif)  
_Sample output with YOLOv8 detecting objects in real time._

## Getting Started 💻

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites
Make sure you have the following installed on your system:
- Python 3.7+
- pip (Python package installer)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/yolo-flask-opencv-app.git
   cd yolo-flask-opencv-app
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model weights (`yolov8m.pt`):
   - You can download the model from the official [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) repository or use your own trained weights.

4. Ensure that your webcam is connected (or use a video feed).

### Running the App

1. Start the Flask web server:
   ```bash
   python app.py
   ```

2. Open a browser and navigate to `http://127.0.0.1:5000/` to view the real-time object detection.

## Project Structure

```
├── app.py               # Main Flask app with object detection logic
├── templates
│   └── index.html       # Front-end HTML template
├── static
│   └── styles.css       # Optional CSS for custom styling
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## YOLOv8 Model 🎯

The project uses the YOLOv8 model for object detection. You can replace `yolov8m.pt` with another YOLOv8 model weight if needed.

### Supported Classes
By default, the following object classes are detected:
- Person
- Car
- Bicycle
- Motorbike
- Bus
- ... and many more (complete list in `app.py`)

## Customization 🛠️

### Changing the Video Source
- Modify `cap = cv2.VideoCapture(0)` in `gen_frames()` function to change the video input. For example, you can provide a video file path instead of `0` for a local webcam feed.

### Adding Custom Models
- You can train your own YOLOv8 model and use the trained `.pt` file by replacing the default model in the `YOLO("yolov8m.pt")` initialization.

## License 📄
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙌
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Flask](https://flask.palletsprojects.com/)

Feel free to contribute, open issues, or suggest new features! 😊
```
