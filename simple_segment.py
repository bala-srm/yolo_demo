import cv2
from ultralytics import YOLO

# Load the YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Read the image
image = cv2.imread('Zebras.jpg')

# Run inference with segmentation
results = model(image)[0]

# Visualize the results on the image (includes masks)
annotated_image = results.plot()

# Display the image
cv2.imshow('Instance Segmentation', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
