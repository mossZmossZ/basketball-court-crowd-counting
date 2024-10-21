import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt, cm as c
from scipy.ndimage import gaussian_filter
import torchvision.transforms.functional as F
from model import CSRNet
import torch
from torchvision import transforms
import cv2

# Check if CUDA is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the model and move it to the appropriate device
model = CSRNet().to(device)

# Load pre-trained weights
checkpoint = torch.load('weights.pth', map_location=device)
model.load_state_dict(checkpoint)

# Set model to evaluation mode
model.eval()

# Load the image
img_path = "../IMG/IMG_9872.jpg"
img = Image.open(img_path).convert('RGB')

# Transform the image for the model
img_tensor = transform(img).unsqueeze(0).to(device)  # Move image tensor to device

# Enable mixed precision for faster inference on CUDA (optional)
with torch.cuda.amp.autocast(enabled=True):
    output = model(img_tensor)

# Predicted count
predicted_count = int(output.detach().cpu().sum().item())
print("Predicted Count : ", predicted_count)

# Heatmap generation
output_np = output.detach().cpu().numpy().reshape(output.shape[2], output.shape[3])

# Resize the density map to the original image size
density_map = Image.fromarray(output_np)
density_map = density_map.resize((img.width, img.height), Image.BICUBIC)
density_map = np.asarray(density_map)

# Normalize the density map for better visualization
density_map_normalized = density_map / np.max(density_map)

# Threshold for detecting "people"
threshold = 0.3  # Adjust this threshold for your use case

# Binary mask for areas with high density
people_map = density_map_normalized > threshold

# Convert binary map to uint8 for contours detection
people_map_uint8 = (people_map * 255).astype(np.uint8)

# Find contours (possible person regions)
contours, _ = cv2.findContours(people_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert original image to a NumPy array for drawing rectangles
img_with_boxes = np.array(img)

# Loop over contours to draw bounding boxes
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if cv2.contourArea(cnt) > 50:  # Ignore small areas (you can adjust this)
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Red rectangle

# Create space for the text at the top
text_space_height = 50  # Fixed height in pixels for the text area
img_with_boxes_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)

# Create a new image canvas with extra space at the top
new_height = img_with_boxes_bgr.shape[0] + text_space_height
new_width = img_with_boxes_bgr.shape[1]
new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

# Set the background color of the top area (optional: white background)
new_image[:text_space_height, :] = [255, 255, 255]  # White background for the text area

# Copy the original image to the lower part of the new image
new_image[text_space_height:, :] = img_with_boxes_bgr

# Convert back to RGB format for displaying with matplotlib
new_image_rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

# display the image
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(new_image_rgb) # display the image
plt.title('Predicted Count : %i' % predicted_count)
plt.axis('off')
