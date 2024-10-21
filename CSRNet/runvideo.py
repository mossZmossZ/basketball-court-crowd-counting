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

# Image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the model
model = CSRNet()

# Load pre-trained weights
checkpoint = torch.load('weights.pth', map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()  # Set model to evaluation mode for inference

# Open video capture (use 0 for webcam, or specify a video file path)
video_path = "Example1.mp4"  # Replace with 0 for webcam
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video or webcam.")
    exit()

# Desired resolution: 1280x720
desired_width = 1280
desired_height = 720

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit loop if no frames left or error in reading

    # Resize frame to 720p
    frame_resized = cv2.resize(frame, (desired_width, desired_height))

    # Convert frame (OpenCV BGR to RGB for PIL compatibility)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # Transform the frame for the model
    img_tensor = transform(img)

    # Predict density map
    output = model(img_tensor.unsqueeze(0))

    # Predicted count
    predicted_count = int(output.detach().cpu().sum().numpy())
    print("Predicted Count: ", predicted_count)

    # Convert the output to a numpy array
    output_np = output.detach().cpu().numpy().reshape(output.shape[2], output.shape[3])

    # Resize the density map to the original frame size
    density_map = Image.fromarray(output_np)
    density_map = density_map.resize((frame_resized.shape[1], frame_resized.shape[0]), Image.BICUBIC)
    density_map = np.asarray(density_map)

    # Normalize the density map for visualization
    density_map_normalized = density_map / np.max(density_map)

    # Threshold for detecting "people"
    threshold = 0.3  # Adjust this threshold for your use case

    # Binary mask for areas with high density
    people_map = density_map_normalized > threshold

    # Convert binary map to uint8 for contours detection
    people_map_uint8 = (people_map * 255).astype(np.uint8)

    # Find contours (possible person regions)
    contours, _ = cv2.findContours(people_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on the frame
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 50:  # Ignore small areas (adjust if needed)
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red rectangle

    # Display the frame with bounding boxes and predicted count
    cv2.putText(frame_resized, f'Count: {predicted_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Crowd Counting', frame_resized)

    # Press 'q' to exit the video
    # Add delay to simulate a lower frame rate (e.g., 30 FPS -> 33 ms delay)
    if cv2.waitKey(200) & 0xFF == ord('q'):  # 100 ms delay slows the frame rate
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
