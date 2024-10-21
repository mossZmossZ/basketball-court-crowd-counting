import cv2
import os
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8x.pt')

# Specify the folder containing images
folder_path = r'../IMG/'  # Adjust the path to your folder
total_people_count = 0  # Counter for total people in all images

# Loop through all image files in the folder
for image_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_name)
    
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to open image at {image_path}. Skipping this file.")
        continue  # Skip to the next image if the file is not valid

    # Perform inference
    results = model(image)

    # Get bounding boxes and labels
    people_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  # 'person' class ID
                people_count += 1
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # xyxy format
                # Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add people count for the current image to the total count
    total_people_count += people_count

    # Resize the image to 769x1024
    resized_image = cv2.resize(image, (944, 1680))

    # Increase label size and display people count on the resized image
    label = f'People: {people_count}'
    cv2.putText(resized_image, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Larger font and thickness

    # Show the resized image
    cv2.imshow(f'People Counting - {image_name}', resized_image)
    cv2.waitKey(1000)  # Display each image for 1 second
    cv2.destroyAllWindows()

# Display the total count after all images are processed
print(f"Total people counted in folder '{folder_path}': {total_people_count}")
