from ultralytics import YOLO
import json

# Predefined COCO class names
coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Load the YOLOv11 model with error handling
try:
    model = YOLO("yolo11n.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Run inference on an image
image_path = "image.jpg"
try:
    results = model(image_path)
except Exception as e:
    print(f"Error processing image: {e}")
    exit()

# Extract the label/class id, confidence, and bounding boxes
labels = results[0].boxes.cls
confidences = results[0].boxes.conf
bboxes = results[0].boxes.xywh

# Set a confidence threshold
threshold = 0.5  # Adjust as necessary

# Prepare a list for storing detections
detections = []

# Loop through detected objects and filter by confidence
for label, confidence, bbox in zip(labels, confidences, bboxes):
    if confidence > threshold:
        class_name = coco_names[int(label)]  # Use the class names list
        print(f"Label: {class_name}, Confidence: {confidence:.2f}, Bbox: {bbox.tolist()}")

        # Append detection to the list
        detections.append({
            "class": class_name,
            "confidence": confidence.item(),
            "bbox": bbox.tolist()
        })

# Save results to a JSON file
with open("detections.json", "w") as f:
    json.dump(detections, f, indent=4)

# Save the results image
results[0].save("results.jpg")

# Optionally display results
# results[0].show()
