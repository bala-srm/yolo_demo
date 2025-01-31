import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the webcam (0 is usually the default webcam)
video = cv2.VideoCapture(0)

# Loop through the video frames
while video.isOpened():
    # Read a frame from the webcam
    success, frame = video.read()
    
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)[0]

        # Visualize the results on the frame
        annotated_frame = results.plot()

        # Display the annotated frame
        cv2.imshow('Webcam Object Detection', annotated_frame)

        # Break the loop if ESC is pressed
        if cv2.waitKey(1) == 27:
            break
    else:
        # Break the loop if the webcam has an error
        break

# Release everything
video.release()
cv2.destroyAllWindows()
