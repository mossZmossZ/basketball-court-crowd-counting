import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.ops import nms

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load a pretrained Mask R-CNN model from torchvision
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)  # Move model to GPU
model.eval()  # Set the model to evaluation mode

# Load the image (replace 'image_path' with your image file)
image_path = '../IMG/IMG_9872.jpg'
image = Image.open(image_path)

# Preprocess the image for the model
transform = T.Compose([T.ToTensor()])  # Convert PIL image to tensor
image_tensor = transform(image).to(device)  # Move the image tensor to the GPU

# Run the image through the model
with torch.no_grad():
    prediction = model([image_tensor])  # Forward pass on GPU

# Extract the predicted bounding boxes, labels, and scores directly on GPU
boxes = prediction[0]['boxes']  # These are already on the GPU
labels = prediction[0]['labels']
scores = prediction[0]['scores']
masks = prediction[0]['masks']  # Keep everything on the GPU

# Apply NMS to filter out overlapping boxes (IoU threshold = 0.5), keeping it on GPU
iou_threshold = 0.5
keep = nms(boxes, scores, iou_threshold)

# Filter results based on NMS, directly on the GPU
boxes = boxes[keep]
labels = labels[keep]
scores = scores[keep]

# Define a minimum confidence threshold, work fully on GPU
confidence_threshold = 0.5
person_label = 1  # COCO label for "person"
people_count = torch.sum((labels == person_label) & (scores > confidence_threshold))

# Output the result
print(f"Number of people detected: {people_count.item()}")

# Optionally: Move data back to CPU for visualization
boxes = boxes.cpu()
labels = labels.cpu()
scores = scores.cpu()

# Display the image with detected bounding boxes (optional)
plt.figure(figsize=(12, 12))
plt.imshow(image)
ax = plt.gca()

# Plot the bounding boxes for each detected person
for i, box in enumerate(boxes):
    if labels[i] == person_label and scores[i] > confidence_threshold:
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, color='red', linewidth=2)
        ax.add_patch(rect)

plt.show()
