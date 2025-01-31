import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video = cv2.VideoCapture('dogs_playing.mp4')

# Loop through the video frames
while video.isOpened():
    # Read a frame from the video
    success, frame = video.read()
    
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)[0]

        # Visualize the results on the frame
        annotated_frame = results.plot()

        # Display the annotated frame
        cv2.imshow('Video Object Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the video is finished
        break

# Release everything
video.release()
cv2.destroyAllWindows()
