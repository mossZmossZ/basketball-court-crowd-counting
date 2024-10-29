
# Basketball Court Crowd Counting

This project leverages computer vision and deep learning to count people on a basketball court in images. Using YOLO and other detection models, it provides visual bounding boxes around detected individuals, making it a valuable tool for learning image analysis and object detection.

## Features

- **Accurate Crowd Detection**: Uses advanced deep learning models to detect and count people in images.
- **Visual Annotations**: Displays bounding boxes around each detected individual.

## Prerequisites

- Python 3.8 or higher
- Ensure you have a compatible GPU if processing large numbers of images or for faster processing, though the application also runs on CPU.
  
## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/basketball-court-crowd-counting.git
   cd basketball-court-crowd-counting
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   
**Section 4: File Structure and Usage**

## File Structure

- `GUI/app.py`: Main file to launch the application.
- `GUI/`: Contains the graphical user interface files and additional modules.
- `requirements.txt`: Lists the dependencies for easy setup.

## Usage

1. **Run the Application**
   - Start the application from the `GUI` folder:
     ```bash
     python GUI/app.py
     ```

2. **Using the GUI**
   - Upon launching, the GUI will provide options to select images for crowd counting.
   - Detected individuals are highlighted with bounding boxes, and the count is displayed.

## Technical Details

- **Programming Language**: Python
- **Libraries**: OpenCV, PyTorch, Matplotlib, PyQt5, PIL, and Ultralytics YOLO
- **Deep Learning Model**: YOLO, with optional integration of other models for object detection.

## Troubleshooting

- Ensure Python 3.8+ is installed.
- Confirm all dependencies are installed by running:
  ```bash
  pip install -r requirements.txt

---

**Section 6: Acknowledgements**

## Acknowledgements

Thanks to the open-source libraries and contributors that make this project possible, including OpenCV, PyTorch, and the Ultralytics YOLO team.
