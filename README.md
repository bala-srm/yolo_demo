# YOLO Object Detection and Segmentation Demo

This project demonstrates object detection and segmentation capabilities using the YOLO (You Only Look Once) model on both images and videos.

## Setup Instructions

1. Create a virtual environment:
```bash
python3 -m venv yolo_demo_venv
```

2. Activate the virtual environment:
- On macOS/Linux:
```bash
source yolo_demo_venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
- `requirements.txt`: Contains all required Python dependencies
- `detect_image.py`: Script for object detection on images
- `detect_video.py`: Script for object detection on video streams

## Features
- Object detection on images
- Object detection on videos
- Instance segmentation
- Real-time detection capabilities
