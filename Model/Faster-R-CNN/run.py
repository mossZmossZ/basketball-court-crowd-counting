import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load a pretrained Faster R-CNN model using the new "weights" argument
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# Load a pretrained model with the updated method
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Load and preprocess the image
image_path = '../IMG/IMG_9872.jpg'
image = Image.open(image_path)
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

# Set a confidence threshold
confidence_threshold = 0.7

# Run inference
with torch.no_grad():
    predictions = model(image_tensor)

# Extract boxes, labels, and scores from the prediction
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# Filter for "person" detections (label 1 in COCO) and apply confidence threshold
person_filter = (labels == 1) & (scores > confidence_threshold)
people_boxes = boxes[person_filter]
people_scores = scores[person_filter]  # Filter scores to match people_boxes

# Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
from torchvision.ops import nms

iou_threshold = 0.3  # Intersection over Union (IoU) threshold for NMS
keep = nms(people_boxes, people_scores, iou_threshold)
filtered_people_boxes = people_boxes[keep]

# Count the remaining people detections
people_count = len(filtered_people_boxes)

print(f"Number of people detected after tuning: {people_count}")

# Show the image with bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(image)

# Draw bounding boxes
for box in filtered_people_boxes:
    x_min, y_min, x_max, y_max = box.tolist()
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()
