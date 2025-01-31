import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Read the image
image = cv2.imread('Zebras.jpg')

# Run inference
results = model(image)[0]

# Visualize the results on the image
annotated_image = results.plot()

# Display the image
cv2.imshow('Object Detection', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
