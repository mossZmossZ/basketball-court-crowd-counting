import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8x.pt')

# Load the image
image_path = r'../IMG/20240926_122416.jpg'  # Adjust the path
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to open image at {image_path}. Please check the file path.")
else:
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

    # Resize the image to 769x1024
    resized_image = cv2.resize(image, (944, 1680))

    # Increase label size and display people count on the resized image
    label = f'People: {people_count}'
    cv2.putText(resized_image, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Larger font and thickness

    # Show the resized image
    cv2.imshow('People Counting', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the resized image
    #cv2.imwrite('resized_image.jpg', resized_image)
