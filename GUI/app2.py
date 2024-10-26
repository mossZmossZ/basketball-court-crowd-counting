import os
import sqlite3
import sys
import cv2
import time
import numpy as np
import torch
import torchvision
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QTableView, QHeaderView, QMessageBox , QGraphicsPixmapItem, QGraphicsScene
from PyQt5.QtGui import QStandardItemModel, QStandardItem , QPixmap 
from PyQt5.QtCore import Qt , pyqtSignal
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.ops import nms
from torchvision import transforms as T

class WorkerMaskRCNN(QtCore.QObject):
    progress = pyqtSignal(int, str, str, int, str)  # Signal to update progress
    completed = pyqtSignal(str)                     # Signal for completion message

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()
        self.total_images = 0

    @QtCore.pyqtSlot()
    def processImages(self):
        conn = sqlite3.connect('image_metadata.db')
        cursor = conn.cursor()

        # Ensure 'model' column exists
        cursor.execute("PRAGMA table_info(image_metadata)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'model' not in columns:
            cursor.execute("ALTER TABLE image_metadata ADD COLUMN model TEXT")
            conn.commit()

        # Determine the first available model in priority order
        cursor.execute("SELECT DISTINCT model FROM image_metadata WHERE model IN ('YOLOv5', 'YOLOv8', 'Faster-RCNN') OR model IS NULL")
        available_models = [row[0] for row in cursor.fetchall()]

        # Decide which model to process based on availability in priority order
        if 'YOLOv5' in available_models:
            base_model = 'YOLOv5'
        elif 'YOLOv8' in available_models:
            base_model = 'YOLOv8'
        elif 'Faster-RCNN' in available_models:
            base_model = 'Faster-R-CNN'
        elif None in available_models:
            base_model = None
        else:
            self.completed.emit("No images to process.")
            conn.close()
            return

        # Select images based on the chosen model or NULL
        if base_model:
            cursor.execute("SELECT id, date, time, image_path FROM image_metadata WHERE model = ?", (base_model,))
        else:
            cursor.execute("SELECT id, date, time, image_path FROM image_metadata WHERE model IS NULL")

        images = cursor.fetchall()

        self.total_images = len(images)
        if self.total_images == 0:
            self.completed.emit(f"No images to process for base model '{base_model}'.")
            return

        output_folder = os.path.join(os.getcwd(), 'PredictIMG/maskrcnn')
        os.makedirs(output_folder, exist_ok=True)

        start_time = time.time()
        times_per_image = []

        for index, (image_id, date, image_time, image_path) in enumerate(images):
            try:
                image_start_time = time.time()

                # Load and preprocess the image
                original_image = Image.open(image_path)
                original_image = ImageOps.exif_transpose(original_image)

                # Resize for memory efficiency
                image = original_image.resize((800, 800))

                # Convert to tensor and run Mask R-CNN model
                transform = T.Compose([T.ToTensor()])
                image_tensor = transform(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    prediction = self.model(image_tensor)

                # Filter results for persons
                boxes = prediction[0]['boxes']
                labels = prediction[0]['labels']
                scores = prediction[0]['scores']
                
                confidence_threshold = 0.3  # Lower threshold to include more detections
                person_label = 1  # Ensure this is the correct class ID

                iou_threshold = 0.4  # Higher IoU threshold
                keep = nms(boxes, scores, iou_threshold)
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]

                people_count = torch.sum((labels == person_label) & (scores > confidence_threshold)).item()

                # Draw bounding boxes on the original image
                image_cv2 = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                scale_x = original_image.width / 800
                scale_y = original_image.height / 800
                
                for i, box in enumerate(boxes):
                    if labels[i] == person_label and scores[i] > confidence_threshold:
                        xmin, ymin, xmax, ymax = box.int().tolist()
                        # Scale bounding box back to original image dimensions
                        xmin = int(xmin * scale_x)
                        ymin = int(ymin * scale_y)
                        xmax = int(xmax * scale_x)
                        ymax = int(ymax * scale_y)
                        cv2.rectangle(image_cv2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Save the predicted image
                predict_image_path = os.path.join(output_folder, f"predict_{os.path.basename(image_path)}")
                cv2.imwrite(predict_image_path, image_cv2)

                # Insert a new row for Mask R-CNN processing of this image
                cursor.execute('''
                    INSERT INTO image_metadata (date, time, people_count, image_path, predict_path, model)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (date, image_time, people_count, image_path, predict_image_path, 'Mask-R-CNN'))
                conn.commit()

                # Track time and emit progress
                image_end_time = time.time()
                times_per_image.append(image_end_time - image_start_time)

                avg_time_per_image = sum(times_per_image) / len(times_per_image)
                remaining_time = avg_time_per_image * (self.total_images - (index + 1))
                remaining_min, remaining_sec = divmod(int(remaining_time), 60)
                remaining_time_str = f"{remaining_min}:{remaining_sec:02d}"

                progress_percent = int((index + 1) / self.total_images * 100)
                self.progress.emit(progress_percent, image_path, predict_image_path, people_count, remaining_time_str)

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

        total_time = time.time() - start_time
        total_min, total_sec = divmod(int(total_time), 60)
        total_time_str = f"{total_min}:{total_sec:02d}"

        self.completed.emit(f"Process completed in {total_time_str} min")
        conn.close()        


class WorkerFasterRCNN(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, str, str, int, str)  # Signal to update progress
    completed = QtCore.pyqtSignal(str)                     # Signal for completion message

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
        self.model.eval()
        self.total_images = 0

    @QtCore.pyqtSlot()
    def processImages(self):
        conn = sqlite3.connect('image_metadata.db')
        cursor = conn.cursor()

        # Ensure 'model' column exists
        cursor.execute("PRAGMA table_info(image_metadata)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'model' not in columns:
            cursor.execute("ALTER TABLE image_metadata ADD COLUMN model TEXT")
            conn.commit()

        # Determine the available models in the database
        cursor.execute("SELECT DISTINCT model FROM image_metadata WHERE model IN ('YOLOv5', 'YOLOv8', 'Mask-R-CNN') OR model IS NULL")
        available_models = [row[0] for row in cursor.fetchall()]

        # Select model preference order
        if 'YOLOv5' in available_models:
            base_model = 'YOLOv5'
        elif 'YOLOv8' in available_models:
            base_model = 'YOLOv8'
        elif 'Mask-R-CNN' in available_models:
            base_model = 'Mask-R-CNN'
        elif None in available_models:
            base_model = None
        else:
            self.completed.emit("No images available to process.")
            conn.close()
            return

        # Select images based on the chosen model or NULL
        if base_model:
            cursor.execute("SELECT id, date, time, image_path FROM image_metadata WHERE model = ?", (base_model,))
        else:
            cursor.execute("SELECT id, date, time, image_path FROM image_metadata WHERE model IS NULL")
        
        images = cursor.fetchall()

        self.total_images = len(images)
        if self.total_images == 0:
            self.completed.emit(f"No images to process for base model '{base_model}'.")
            return

        output_folder = os.path.join(os.getcwd(), 'PredictIMG/fasterrcnn')
        os.makedirs(output_folder, exist_ok=True)

        start_time = time.time()
        times_per_image = []

        for index, (image_id, date, image_time, image_path) in enumerate(images):
            try:
                image_start_time = time.time()

                # Load the original image and preprocess
                original_image = Image.open(image_path)
                original_image = ImageOps.exif_transpose(original_image)
                
                # Calculate scale factors
                scale_x = original_image.width / 800
                scale_y = original_image.height / 800

                # Resize for model input and convert to tensor
                resized_image = original_image.resize((800, 800))
                transform = transforms.Compose([transforms.ToTensor()])
                image_tensor = transform(resized_image).unsqueeze(0).to(self.device)

                # Run Faster R-CNN prediction
                with torch.no_grad():
                    predictions = self.model(image_tensor)

                # Filter for persons (label 1 in COCO)
                boxes = predictions[0]['boxes']
                labels = predictions[0]['labels']
                scores = predictions[0]['scores']
                
                confidence_threshold = 0.5
                person_label = 1  # COCO label for "person"
                person_filter = (labels == person_label) & (scores > confidence_threshold)
                
                people_boxes = boxes[person_filter]
                people_scores = scores[person_filter]

                # Apply NMS
                iou_threshold = 0.3
                keep = nms(people_boxes, people_scores, iou_threshold)
                filtered_boxes = people_boxes[keep]

                # Count people and draw bounding boxes on the original image
                people_count = len(filtered_boxes)
                image_cv2 = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                for box in filtered_boxes:
                    # Scale bounding box back to original image dimensions
                    x_min, y_min, x_max, y_max = box.tolist()
                    x_min = int(x_min * scale_x)
                    y_min = int(y_min * scale_y)
                    x_max = int(x_max * scale_x)
                    y_max = int(y_max * scale_y)
                    cv2.rectangle(image_cv2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Save the predicted image with bounding boxes
                predict_image_path = os.path.join(output_folder, f"predict_{os.path.basename(image_path)}")
                cv2.imwrite(predict_image_path, image_cv2)

                # Insert a new row with model set to 'Faster-RCNN'
                cursor.execute('''
                    INSERT INTO image_metadata (date, time, people_count, image_path, predict_path, model)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (date, image_time, people_count, image_path, predict_image_path, 'Faster-R-CNN'))
                conn.commit()

                # Track time and emit progress
                image_end_time = time.time()
                times_per_image.append(image_end_time - image_start_time)

                avg_time_per_image = sum(times_per_image) / len(times_per_image)
                remaining_time = avg_time_per_image * (self.total_images - (index + 1))
                remaining_min, remaining_sec = divmod(int(remaining_time), 60)
                remaining_time_str = f"{remaining_min}:{remaining_sec:02d}"

                progress_percent = int((index + 1) / self.total_images * 100)
                self.progress.emit(progress_percent, image_path, predict_image_path, people_count, remaining_time_str)

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue


        total_time = time.time() - start_time
        total_min, total_sec = divmod(int(total_time), 60)
        total_time_str = f"{total_min}:{total_sec:02d}"

        self.completed.emit(f"Process completed in {total_time_str} min")
        conn.close()


class WorkerYoloV8(QtCore.QObject):
    # Other parts of the class remain the same
    progress = QtCore.pyqtSignal(int, str, str, int, str)  # Signal to update progress (progress, original image path, predicted image path, people_count, remaining_time)
    completed = QtCore.pyqtSignal(str)                     # Signal for completion message

    def __init__(self):
        super().__init__()
        self.model = YOLO('yolov8x.pt')
        self.total_images = 0  # Initialize the total_images variable


    @QtCore.pyqtSlot()
    def processImages(self):
        conn = sqlite3.connect('image_metadata.db')
        cursor = conn.cursor()

        # Check if 'model' column exists
        cursor.execute("PRAGMA table_info(image_metadata)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'model' not in columns:
            cursor.execute("ALTER TABLE image_metadata ADD COLUMN model TEXT")
            conn.commit()

        # Select images that haven't been processed by YOLOv8
        cursor.execute("SELECT id, date, time AS image_time, image_path FROM image_metadata WHERE model = 'YOLOv5'")
        images = cursor.fetchall()

        self.total_images = len(images)
        if self.total_images == 0:
            self.completed.emit("No images to process.")
            return

        output_folder = os.path.join(os.getcwd(), 'PredictIMG/YoloV8')
        os.makedirs(output_folder, exist_ok=True)

        start_time = time.time()
        times_per_image = []

        for index, (image_id, date, image_time, image_path) in enumerate(images):
            image_start_time = time.time()

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to open image at {image_path}. Skipping this file.")
                continue

            results = self.model(image)
            people_count = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.cls == 0:
                        people_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

            predict_image_path = os.path.join(output_folder, f"predict_{os.path.basename(image_path)}")
            cv2.imwrite(predict_image_path, image)

            # Insert a new row for YOLOv8 processing of this image
            cursor.execute('''
                INSERT INTO image_metadata (date, time, people_count, image_path, predict_path, model)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (date, image_time, people_count, image_path, predict_image_path, 'YOLOv8'))
            conn.commit()

            image_end_time = time.time()
            time_for_image = image_end_time - image_start_time
            times_per_image.append(time_for_image)

            avg_time_per_image = sum(times_per_image) / len(times_per_image)
            remaining_time = avg_time_per_image * (self.total_images - (index + 1))

            remaining_min, remaining_sec = divmod(int(remaining_time), 60)
            remaining_time_str = f"{remaining_min}:{remaining_sec:02d}"

            progress_percent = int((index + 1) / self.total_images * 100)
            self.progress.emit(progress_percent, image_path, predict_image_path, people_count, remaining_time_str)

        total_time = time.time() - start_time
        total_min, total_sec = divmod(int(total_time), 60)
        total_time_str = f"{total_min}:{total_sec:02d}"

        self.completed.emit(f"Process completed in {total_time_str} min")
        conn.close()



class WorkerYoloV5(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, str, str, int, str)  # Signal to update progress (progress, original image path, predicted image path, people_count, remaining_time)
    completed = QtCore.pyqtSignal(str)                     # Signal for completion message

    def __init__(self):
        super().__init__()
        self.model = YOLO('yolov5x.pt')
        self.total_images = 0  # Initialize the total_images variable

    @QtCore.pyqtSlot()
    def processImages(self):
        # Connect to the SQLite database
        conn = sqlite3.connect('image_metadata.db')
        cursor = conn.cursor()

        # Check if 'model' column exists, if not, add it
        cursor.execute("PRAGMA table_info(image_metadata)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'model' not in columns:
            cursor.execute("ALTER TABLE image_metadata ADD COLUMN model TEXT")
            conn.commit()

        # Select all images from the database
        cursor.execute("SELECT id, image_path FROM image_metadata WHERE predict_path IS NULL")
        images = cursor.fetchall()

        self.total_images = len(images)  # Store the total number of images
        if self.total_images == 0:
            self.completed.emit("No images to process.")
            return

        # Create output folder if not exists
        output_folder = os.path.join(os.getcwd(), 'PredictIMG/YoloV5')
        os.makedirs(output_folder, exist_ok=True)

        # Variables to track time
        start_time = time.time()  # Start time of the entire process
        times_per_image = []

        # Loop through each image in the database
        for index, (image_id, image_path) in enumerate(images):
            image_start_time = time.time()

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to open image at {image_path}. Skipping this file.")
                continue

            # Perform prediction using YOLO
            results = self.model(image)

            # Get bounding boxes and labels
            people_count = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.cls == 0:  # 'person' class ID
                        people_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

            # Save the predicted image
            predict_image_path = os.path.join(output_folder, f"predict_{os.path.basename(image_path)}")
            cv2.imwrite(predict_image_path, image)

            # Update the database with the prediction results and model
            cursor.execute('''
                UPDATE image_metadata
                SET people_count = ?, predict_path = ?, model = ?
                WHERE id = ?
            ''', (people_count, predict_image_path, 'YOLOv5', image_id))
            conn.commit()

            # Stop timer for this image and calculate the time taken
            image_end_time = time.time()
            time_for_image = image_end_time - image_start_time
            times_per_image.append(time_for_image)  # Store the time for this image

            # Estimate remaining time
            avg_time_per_image = sum(times_per_image) / len(times_per_image)
            remaining_time = avg_time_per_image * (self.total_images - (index + 1))

            # Convert remaining time to min:ss format
            remaining_min, remaining_sec = divmod(int(remaining_time), 60)
            remaining_time_str = f"{remaining_min}:{remaining_sec:02d}"

            # Emit progress signal
            progress_percent = int((index + 1) / self.total_images * 100)
            self.progress.emit(progress_percent, image_path, predict_image_path, people_count, remaining_time_str)

        # Total time taken for the process
        total_time = time.time() - start_time

        # Convert total time to min:ss format
        total_min, total_sec = divmod(int(total_time), 60)
        total_time_str = f"{total_min}:{total_sec:02d}"

        self.completed.emit(f"Process completed in {total_time_str} min")
        conn.close()


# Main UI class for Basketball Crowd Counting
class Ui_BasketballCrowdCounting(object):
    def setupUi(self, BasketballCrowdCounting):
        self.main_window = BasketballCrowdCounting  # Store the QMainWindow instance
        BasketballCrowdCounting.setObjectName("BasketballCrowdCounting")
        BasketballCrowdCounting.resize(1366, 768)
        self.centralwidget = QtWidgets.QWidget(BasketballCrowdCounting)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.CountImg = QtWidgets.QLCDNumber(self.centralwidget)
        self.CountImg.setObjectName("CountImg")
        self.gridLayout.addWidget(self.CountImg, 5, 1, 1, 1)
        self.OpenFolderButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(26)
        self.OpenFolderButton.setFont(font)
        self.OpenFolderButton.setObjectName("OpenFolderButton")
        self.OpenFolderButton.clicked.connect(self.openFolderDialog)  # Connect the button to openFolderDialog
        self.gridLayout.addWidget(self.OpenFolderButton, 0, 0, 1, 1)
        self.tableFolder = QtWidgets.QTableView(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.tableFolder.setFont(font)
        self.tableFolder.setObjectName("tableFolder")
        self.gridLayout.addWidget(self.tableFolder, 1, 0, 2, 1)
        self.Number = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Number.setFont(font)
        self.Number.setObjectName("Number")
        self.gridLayout.addWidget(self.Number, 4, 1, 1, 1)
        self.ProcessButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.ProcessButton.setFont(font)
        self.ProcessButton.setObjectName("ProcessButton")
        self.ProcessButton.setEnabled(False)  # Initially disabled
        self.ProcessButton.clicked.connect(lambda: self.openNewPage(BasketballCrowdCounting))  # Pass window instance
        self.gridLayout.addWidget(self.ProcessButton, 5, 0, 1, 1)
        self.FolderLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.FolderLabel.setFont(font)
        self.FolderLabel.setObjectName("FolderLabel")
        self.gridLayout.addWidget(self.FolderLabel, 4, 0, 1, 1)
        self.OpenDatabase = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(26)
        self.OpenDatabase.setFont(font)
        self.OpenDatabase.setObjectName("OpenDatabase")
        self.OpenDatabase.clicked.connect(self.openDatabasePage)
        self.gridLayout.addWidget(self.OpenDatabase, 0, 1, 1, 1)
        self.CreditButton = QtWidgets.QPushButton(self.centralwidget)
        self.CreditButton.setObjectName("CreditButton")
        self.gridLayout.addWidget(self.CreditButton, 6, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.YoloV5_CheckBox = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.YoloV5_CheckBox.setFont(font)
        self.YoloV5_CheckBox.setAutoRepeat(False)
        self.YoloV5_CheckBox.setObjectName("YoloV5_CheckBox")
        self.verticalLayout.addWidget(self.YoloV5_CheckBox)
        self.YoloV8_CheckBox = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.YoloV8_CheckBox.setFont(font)
        self.YoloV8_CheckBox.setObjectName("YoloV8_CheckBox")
        self.verticalLayout.addWidget(self.YoloV8_CheckBox)
        self.MaskRcheckBox = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.MaskRcheckBox.setFont(font)
        self.MaskRcheckBox.setObjectName("MaskRcheckBox")
        self.verticalLayout.addWidget(self.MaskRcheckBox)
        self.FasterRCheckbox = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.FasterRCheckbox.setFont(font)
        self.FasterRCheckbox.setObjectName("FasterRCheckbox")
        self.verticalLayout.addWidget(self.FasterRCheckbox)
        self.gridLayout.addLayout(self.verticalLayout, 2, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 1, 1, 1)
        BasketballCrowdCounting.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(BasketballCrowdCounting)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 21))
        self.menubar.setObjectName("menubar")
        BasketballCrowdCounting.setMenuBar(self.menubar)

        self.retranslateUi(BasketballCrowdCounting)
        QtCore.QMetaObject.connectSlotsByName(BasketballCrowdCounting)

    def retranslateUi(self, BasketballCrowdCounting):
        _translate = QtCore.QCoreApplication.translate
        BasketballCrowdCounting.setWindowTitle(_translate("BasketballCrowdCounting", "MainWindow"))
        self.OpenFolderButton.setText(_translate("BasketballCrowdCounting", "Open Folder"))
        self.Number.setText(_translate("BasketballCrowdCounting", "Total Image"))
        self.ProcessButton.setText(_translate("BasketballCrowdCounting", "Start Process"))
        self.FolderLabel.setText(_translate("BasketballCrowdCounting", "No Folder Selected!"))
        self.OpenDatabase.setText(_translate("BasketballCrowdCounting", "Open Database"))
        self.CreditButton.setText(_translate("BasketballCrowdCounting", "Credit"))
        self.YoloV5_CheckBox.setText(_translate("BasketballCrowdCounting", "YoloV5"))
        self.YoloV8_CheckBox.setText(_translate("BasketballCrowdCounting", "YoloV8"))
        self.MaskRcheckBox.setText(_translate("BasketballCrowdCounting", "Mask-R-CNN"))
        self.FasterRCheckbox.setText(_translate("BasketballCrowdCounting", "Faster-R-CNN"))
        self.label.setText(_translate("BasketballCrowdCounting", "Select Models"))

    def openFolderDialog(self):
        # Open a folder dialog and get the selected folder path
        folder = QFileDialog.getExistingDirectory(None, 'Select Folder')
        if folder:
            self.FolderLabel.setText(f'Selected Folder: {folder}')
            self.selectedFolder = folder
            self.ProcessButton.setEnabled(True)  # Enable the process button
            self.processImages()
            self.showTable(folder)
            self.ProcessButton.setEnabled(True)  # Enable Process Button

    def processImages(self):
        # Ensure a folder is selected
        if not hasattr(self, 'selectedFolder'):
            self.FolderLabel.setText('Please select a folder first!')
            return

        # Path to the selected folder
        folder_path = self.selectedFolder

        # Database setup
        conn = sqlite3.connect('image_metadata.db')  # Connect to SQLite database
        cursor = conn.cursor()

        # Create table if it doesn't exist, adding the new 'model' column
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time TEXT,
                people_count INTEGER DEFAULT NULL,
                image_path TEXT,
                predict_path TEXT DEFAULT NULL,
                model TEXT DEFAULT NULL  -- New column 'model'
            )
        ''')

        image_count = 0

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Filter image files
                image_path = os.path.join(folder_path, filename)
                image_count += 1
                try:
                    # Open the image
                    image = Image.open(image_path)
                    # Extracting the exif metadata
                    exifdata = image.getexif()
                    # Initialize variables for date and time
                    date = None
                    time = None
                    # Loop through all the tags present in exifdata
                    for tagid in exifdata:
                        tagname = TAGS.get(tagid, tagid)
                        value = exifdata.get(tagid)
                        # Looking for the DateTime tag
                        if tagname == 'DateTime':
                            # Original format: 'YYYY:MM:DD HH:MM:SS'
                            datetime_str = value
                            # Converting to datetime object
                            dt_obj = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
                            # Formatting the date as YYYY-MM-DD
                            date = dt_obj.strftime('%Y-%m-%d')
                            # Formatting the time as HH:MM
                            time = dt_obj.strftime('%H:%M')
                    # Insert the extracted data into the database, setting people_count, predict_path, and model as None
                    cursor.execute('''
                        INSERT INTO image_metadata (date, time, people_count, image_path, predict_path, model)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (date, time, None, image_path, None, None))  # 'model' is set to None
                    print(f"Processed: {image_path} - Date: {date}, Time: {time}")
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
        
        # Commit the transaction and close the connection
        conn.commit()
        conn.close()

        # Update the LCD display and label after processing
        self.CountImg.display(image_count)
        self.FolderLabel.setText(f'Images loaded successfully!')
        self.OpenFolderButton.setEnabled(False)  # Disable Open Folder Button
        self.OpenDatabase.setEnabled(False)  # Disable Open Database Button

    def showTable(self, folder):
        # Create a model to hold the data
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(['Path'])  # Only one header for the path

        # Loop through all files in the folder and add them to the model
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Only include image files
                image_path = os.path.join(folder, filename)
                model.appendRow([QStandardItem(image_path)])

        # Set the model to the table view
        self.tableFolder.setModel(model)
        self.tableFolder.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Make the column stretch to fit

    def openNewPage(self, BasketballCrowdCounting):
        selected_models = self.get_selected_models()
        if selected_models:
            self.controller = Controller(BasketballCrowdCounting)  # Pass the QMainWindow here
            self.controller.startProcessing(selected_models)
        else:
            QMessageBox.warning(None, "No Model Selected", "Please select at least one model.")
        
    def openDatabasePage(self):
        # Open a file dialog to select the database file
        db_file, _ = QFileDialog.getOpenFileName(None, "Select Database File", "", "Database Files (*.db)")

        if db_file:
            self.selected_db_file = db_file  # Store the selected file path
        else:
            self.selected_db_file = 'image_metadata.db'  # Default to the internal 'image_metadata.db'

        self.redirectToGraphPage(self.main_window)  # Redirect to the graph page

    def redirectToGraphPage(self, MainWindow):
        self.graph_page_ui = GraphPage()  # Create a new instance of the graph page
        # Pass the main window and the selected (or default) database file path
        self.graph_page_ui.setupUi(MainWindow, self.selected_db_file)

    def get_selected_models(self):
        models = []
        if self.YoloV5_CheckBox.isChecked():
            models.append("YoloV5")
        if self.YoloV8_CheckBox.isChecked():
            models.append("YoloV8")
        if self.MaskRcheckBox.isChecked():
            models.append("Mask-R-CNN")
        if self.FasterRCheckbox.isChecked():
            models.append("Faster-R-CNN")
        return models if models else None
    
# Controller class for handling multiple pages
class Controller(QtWidgets.QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.selected_models = []
        self.current_model_index = 0

    def startProcessing(self, selected_models):
        self.selected_models = selected_models
        self.current_model_index = 0
        self.showNextModelProcess()

    def showNextModelProcess(self):
        # Clear any existing central widget to ensure no overlapping views
        self.main_window.setCentralWidget(None)
        QtWidgets.QApplication.processEvents()  # Force GUI to update

        # Check if we have more models to process
        if self.current_model_index < len(self.selected_models):
            model_name = self.selected_models[self.current_model_index]
            print(f"Processing model: {model_name}")  # Debug statement

            # Initialize and set up the next model processing page
            if model_name == "YoloV5":
                self.ui_yolov5 = YOLOv5ProcessingPage(self)
                self.ui_yolov5.setupUi(self.main_window)
                QtCore.QTimer.singleShot(100, self.ui_yolov5.startProcessing)  # Delay before starting
            elif model_name == "YoloV8":
                self.ui_yolov8 = YOLOv8ProcessingPage(self)
                self.ui_yolov8.setupUi(self.main_window)
                QtCore.QTimer.singleShot(100, self.ui_yolov8.startProcessing)
            elif model_name == "Mask-R-CNN":
                self.ui_maskrcnn = MaskRCNNProcessingPage(self)
                self.ui_maskrcnn.setupUi(self.main_window)
                QtCore.QTimer.singleShot(100, self.ui_maskrcnn.startProcessing)
            elif model_name == "Faster-R-CNN":
                self.ui_fasterrcnn = FasterRCNNProcessingPage(self)
                self.ui_fasterrcnn.setupUi(self.main_window)
                QtCore.QTimer.singleShot(100, self.ui_fasterrcnn.startProcessing)

            # Move to the next model for the subsequent call
            self.current_model_index += 1
        else:
            # If no more models, show the graph page
            print("All models processed. Showing the graph page.")
            self.showGraphPage(self.main_window)

    def showGraphPage(self, main_window):
        self.graph_page = GraphPage()
        self.graph_page.setupUi(main_window, self.selected_db_file if hasattr(self, 'selected_db_file') else None)

class YOLOv5ProcessingPage:
    def __init__(self,  controller):
        self.controller = controller
        
    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.Original_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.Original_label.setFont(font)
        self.Original_label.setObjectName("Original_label")
        self.gridLayout.addWidget(self.Original_label, 0, 0, 1, 1)
        self.Predict_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.Predict_label.setFont(font)
        self.Predict_label.setObjectName("Predict_label")
        self.gridLayout.addWidget(self.Predict_label, 0, 1, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout.addLayout(self.verticalLayout, 0, 2, 1, 1)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 1)
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout.addWidget(self.graphicsView_2, 1, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.People_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.People_label.setFont(font)
        self.People_label.setObjectName("People_label")
        self.horizontalLayout.addWidget(self.People_label)
        self.lcd_count = QtWidgets.QLCDNumber(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.lcd_count.setFont(font)
        self.lcd_count.setObjectName("lcd_count")
        self.horizontalLayout.addWidget(self.lcd_count)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 1, 1, 1)
        self.Remaining_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.Remaining_label.setFont(font)
        self.Remaining_label.setObjectName("Remaining_label")
        self.gridLayout.addWidget(self.Remaining_label, 4, 0, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.progressBar.setFont(font)
        self.progressBar.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 4, 1, 1, 1)
        self.process_text = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.process_text.setFont(font)
        self.process_text.setObjectName("process_text")
        self.gridLayout.addWidget(self.process_text, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1366, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Create a QThread and a worker
        self.thread = QtCore.QThread()
        self.worker = WorkerYoloV5()

        # Move the worker to the thread
        self.worker.moveToThread(self.thread)

        # Connect the worker's signals to the UI
        self.worker.progress.connect(self.updateProgress)
        self.worker.completed.connect(self.showCompletionMessage)

        # Connect the button for starting the process
        self.startProcessing()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Original_label.setText(_translate("MainWindow", "Original Image"))
        self.Predict_label.setText(_translate("MainWindow", "YoloV5Predict Image"))
        self.People_label.setText(_translate("MainWindow", "People Count :"))
        self.Remaining_label.setText(_translate("MainWindow", "Time Remaining :"))
        self.process_text.setText(_translate("MainWindow", "Processing :"))

    def startProcessing(self):
        self.thread.started.connect(self.worker.processImages)
        self.thread.start()

    def showCompletionMessage(self, message):
        self.process_text.setText(message)
        self.thread.quit()
        QtCore.QTimer.singleShot(1000, self.controller.showNextModelProcess)  # Simulate 2 seconds processing


    def updateProgress(self, progress, original_path, predict_image_path, people_count, remaining_time_str):
        # Update progress bar, people count, and display images
        self.progressBar.setValue(progress)
        self.lcd_count.display(people_count)
        
        # Display remaining time in the label
        self.Remaining_label.setText(f"Time remaining: {remaining_time_str} min")
        
        # Calculate and display the number of processed images out of total images
        total_images = self.worker.total_images
        current_image = int(progress / 100 * total_images)
        
        # Update the process text with the current image number and total
        self.process_text.setText(f"Processing image {current_image + 1} of {total_images}...")  # e.g., "Processing image 1 of 100"

        # Display original image and predicted image
        self.displayImage(original_path, self.graphicsView)  # Display original image
        self.displayImage(predict_image_path, self.graphicsView_2)  # Display predicted image

    def displayImage(self, img_path, graphicsView):
        scene = QGraphicsScene()
        pixmap = QPixmap(img_path)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        graphicsView.setScene(scene)
        graphicsView.fitInView(item, QtCore.Qt.KeepAspectRatioByExpanding)
        graphicsView.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)


# YOLOv8 Processing Page
class YOLOv8ProcessingPage:
    def __init__(self,  controller):
        self.controller = controller
        
    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.Original_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.Original_label.setFont(font)
        self.Original_label.setObjectName("Original_label")
        self.gridLayout.addWidget(self.Original_label, 0, 0, 1, 1)
        self.Predict_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.Predict_label.setFont(font)
        self.Predict_label.setObjectName("Predict_label")
        self.gridLayout.addWidget(self.Predict_label, 0, 1, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout.addLayout(self.verticalLayout, 0, 2, 1, 1)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 1)
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout.addWidget(self.graphicsView_2, 1, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.People_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.People_label.setFont(font)
        self.People_label.setObjectName("People_label")
        self.horizontalLayout.addWidget(self.People_label)
        self.lcd_count = QtWidgets.QLCDNumber(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.lcd_count.setFont(font)
        self.lcd_count.setObjectName("lcd_count")
        self.horizontalLayout.addWidget(self.lcd_count)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 1, 1, 1)
        self.Remaining_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.Remaining_label.setFont(font)
        self.Remaining_label.setObjectName("Remaining_label")
        self.gridLayout.addWidget(self.Remaining_label, 4, 0, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.progressBar.setFont(font)
        self.progressBar.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 4, 1, 1, 1)
        self.process_text = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.process_text.setFont(font)
        self.process_text.setObjectName("process_text")
        self.gridLayout.addWidget(self.process_text, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1366, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Create a QThread and a worker
        self.thread = QtCore.QThread()
        self.worker = WorkerYoloV8()

        # Move the worker to the thread
        self.worker.moveToThread(self.thread)

        # Connect the worker's signals to the UI
        self.worker.progress.connect(self.updateProgress)
        self.worker.completed.connect(self.showCompletionMessage)

        # Connect the button for starting the process
        self.startProcessing()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Original_label.setText(_translate("MainWindow", "Original Image"))
        self.Predict_label.setText(_translate("MainWindow", "YoloV8 Predict Image"))
        self.People_label.setText(_translate("MainWindow", "People Count :"))
        self.Remaining_label.setText(_translate("MainWindow", "Time Remaining :"))
        self.process_text.setText(_translate("MainWindow", "Processing :"))


    def startProcessing(self):
        self.thread.started.connect(self.worker.processImages)
        self.thread.start()

    def showCompletionMessage(self, message):
        self.process_text.setText(message)
        self.thread.quit()
        QtCore.QTimer.singleShot(1000, self.controller.showNextModelProcess)  # Simulate 2 seconds processing

    def updateProgress(self, progress, original_path, predict_image_path, people_count, remaining_time_str):
        # Update progress bar, people count, and display images
        self.progressBar.setValue(progress)
        self.lcd_count.display(people_count)
        
        # Display remaining time in the label
        self.Remaining_label.setText(f"Time remaining: {remaining_time_str} min")
        
        # Calculate and display the number of processed images out of total images
        total_images = self.worker.total_images
        current_image = int(progress / 100 * total_images)
        
        # Update the process text with the current image number and total
        self.process_text.setText(f"Processing image {current_image + 1} of {total_images}...")  # e.g., "Processing image 1 of 100"

        # Display original image and predicted image
        self.displayImage(original_path, self.graphicsView)  # Display original image
        self.displayImage(predict_image_path, self.graphicsView_2)  # Display predicted image

    def displayImage(self, img_path, graphicsView):
        scene = QGraphicsScene()
        pixmap = QPixmap(img_path)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        graphicsView.setScene(scene)
        graphicsView.fitInView(item, QtCore.Qt.KeepAspectRatioByExpanding)
        graphicsView.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)



class MaskRCNNProcessingPage:
    def __init__(self, controller):
        self.controller = controller

    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.Original_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.Original_label.setFont(font)
        self.Original_label.setObjectName("Original_label")
        self.gridLayout.addWidget(self.Original_label, 0, 0, 1, 1)
        self.Predict_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.Predict_label.setFont(font)
        self.Predict_label.setObjectName("Predict_label")
        self.gridLayout.addWidget(self.Predict_label, 0, 1, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout.addLayout(self.verticalLayout, 0, 2, 1, 1)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 1)
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout.addWidget(self.graphicsView_2, 1, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.People_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.People_label.setFont(font)
        self.People_label.setObjectName("People_label")
        self.horizontalLayout.addWidget(self.People_label)
        self.lcd_count = QtWidgets.QLCDNumber(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.lcd_count.setFont(font)
        self.lcd_count.setObjectName("lcd_count")
        self.horizontalLayout.addWidget(self.lcd_count)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 1, 1, 1)
        self.Remaining_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.Remaining_label.setFont(font)
        self.Remaining_label.setObjectName("Remaining_label")
        self.gridLayout.addWidget(self.Remaining_label, 4, 0, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.progressBar.setFont(font)
        self.progressBar.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 4, 1, 1, 1)
        self.process_text = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.process_text.setFont(font)
        self.process_text.setObjectName("process_text")
        self.gridLayout.addWidget(self.process_text, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1366, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Create a QThread and a worker
        self.thread = QtCore.QThread()
        self.worker = WorkerMaskRCNN()

        # Move the worker to the thread
        self.worker.moveToThread(self.thread)

        # Connect the worker's signals to the UI
        self.worker.progress.connect(self.updateProgress)
        self.worker.completed.connect(self.showCompletionMessage)

        # Connect the button for starting the process
        self.startProcessing()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Original_label.setText(_translate("MainWindow", "Original Image"))
        self.Predict_label.setText(_translate("MainWindow", "Mask-R-CNN Predict Image"))
        self.People_label.setText(_translate("MainWindow", "People Count :"))
        self.Remaining_label.setText(_translate("MainWindow", "Time Remaining :"))
        self.process_text.setText(_translate("MainWindow", "Processing :"))

    def startProcessing(self):
        self.thread.started.connect(self.worker.processImages)
        self.thread.start()

    def updateProgress(self, progress, original_path, predict_image_path, people_count, remaining_time_str):
        # Update progress bar, people count, and display images
        self.progressBar.setValue(progress)
        self.lcd_count.display(people_count)
        
        # Display remaining time in the label
        self.Remaining_label.setText(f"Time remaining: {remaining_time_str} min")
        
        # Calculate and display the number of processed images out of total images
        total_images = self.worker.total_images
        current_image = int(progress / 100 * total_images)

        # Update the process text with the current image number and total
        self.process_text.setText(f"Processing image {current_image + 1} of {total_images}...")  # e.g., "Processing image 1 of 100"

        # Display original image and predicted image
        self.displayImage(original_path, self.graphicsView)  # Display original image
        self.displayImage(predict_image_path, self.graphicsView_2)  # Display predicted image


    def showCompletionMessage(self, message):
        self.process_text.setText(message)
        self.thread.quit()
        QtCore.QTimer.singleShot(1000, self.controller.showNextModelProcess)  # Simulate 2 seconds processing

    def displayImage(self, img_path, graphicsView):
        # Load and correct the orientation of the image
        original_image = Image.open(img_path)
        original_image = ImageOps.exif_transpose(original_image)  # Correct orientation

        # Convert to QPixmap for display in PyQt5
        img_data = original_image.convert("RGB")
        data = img_data.tobytes("raw", "RGB")
        qimage = QtGui.QImage(data, img_data.width, img_data.height, QtGui.QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # Display the corrected image in the QGraphicsView
        scene = QGraphicsScene()
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        graphicsView.setScene(scene)
        graphicsView.fitInView(item, QtCore.Qt.KeepAspectRatioByExpanding)
        graphicsView.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

class FasterRCNNProcessingPage:
    def __init__(self, controller):
        self.controller = controller

    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.Original_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.Original_label.setFont(font)
        self.Original_label.setObjectName("Original_label")
        self.gridLayout.addWidget(self.Original_label, 0, 0, 1, 1)
        self.Predict_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.Predict_label.setFont(font)
        self.Predict_label.setObjectName("Predict_label")
        self.gridLayout.addWidget(self.Predict_label, 0, 1, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout.addLayout(self.verticalLayout, 0, 2, 1, 1)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 1)
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout.addWidget(self.graphicsView_2, 1, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.People_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.People_label.setFont(font)
        self.People_label.setObjectName("People_label")
        self.horizontalLayout.addWidget(self.People_label)
        self.lcd_count = QtWidgets.QLCDNumber(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.lcd_count.setFont(font)
        self.lcd_count.setObjectName("lcd_count")
        self.horizontalLayout.addWidget(self.lcd_count)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 1, 1, 1)
        self.Remaining_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.Remaining_label.setFont(font)
        self.Remaining_label.setObjectName("Remaining_label")
        self.gridLayout.addWidget(self.Remaining_label, 4, 0, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.progressBar.setFont(font)
        self.progressBar.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 4, 1, 1, 1)
        self.process_text = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.process_text.setFont(font)
        self.process_text.setObjectName("process_text")
        self.gridLayout.addWidget(self.process_text, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1366, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Create a QThread and a worker
        self.thread = QtCore.QThread()
        self.worker = WorkerFasterRCNN()

        # Move the worker to the thread
        self.worker.moveToThread(self.thread)

        # Connect the worker's signals to the UI
        self.worker.progress.connect(self.updateProgress)
        self.worker.completed.connect(self.showCompletionMessage)

        # Connect the button for starting the process
        self.startProcessing()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Original_label.setText(_translate("MainWindow", "Original Image"))
        self.Predict_label.setText(_translate("MainWindow", "Faster-R-CNN Predict Image"))
        self.People_label.setText(_translate("MainWindow", "People Count :"))
        self.Remaining_label.setText(_translate("MainWindow", "Time Remaining :"))
        self.process_text.setText(_translate("MainWindow", "Processing :"))


    def startProcessing(self):
        self.thread.started.connect(self.worker.processImages)
        self.thread.start()

    def updateProgress(self, progress, original_path, predict_image_path, people_count, remaining_time_str):
        # Update progress bar, people count, and display images
        self.progressBar.setValue(progress)
        self.lcd_count.display(people_count)
        
        # Display remaining time in the label
        self.Remaining_label.setText(f"Time remaining: {remaining_time_str} min")
        
        # Calculate and display the number of processed images out of total images
        total_images = self.worker.total_images
        current_image = int(progress / 100 * total_images)
        
        # Update the process text with the current image number and total
        self.process_text.setText(f"Processing image {current_image + 1} of {total_images}...")  # e.g., "Processing image 1 of 100"

        # Display original image and predicted image
        self.displayImage(original_path, self.graphicsView)  # Display original image
        self.displayImage(predict_image_path, self.graphicsView_2)  # Display predicted image


    def showCompletionMessage(self, message):
        self.process_text.setText(message)
        self.thread.quit()
        QtCore.QTimer.singleShot(1000, self.controller.showNextModelProcess)  # Simulate 2 seconds processing

    def displayImage(self, img_path, graphicsView):
        # Load and correct the orientation of the image
        original_image = Image.open(img_path)
        original_image = ImageOps.exif_transpose(original_image)  # Correct orientation

        # Convert to QPixmap for display in PyQt5
        img_data = original_image.convert("RGB")
        data = img_data.tobytes("raw", "RGB")
        qimage = QtGui.QImage(data, img_data.width, img_data.height, QtGui.QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # Display the corrected image in the QGraphicsView
        scene = QGraphicsScene()
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        graphicsView.setScene(scene)
        graphicsView.fitInView(item, QtCore.Qt.KeepAspectRatioByExpanding)
        graphicsView.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)


class GraphPage:
    def __init__(self):
        pass  # No arguments needed

    def setupUi(self, MainWindow, db_file_path=None):
        MainWindow.setObjectName("GraphPage")
        MainWindow.resize(1366, 768)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Show the selected or default database path
        if not db_file_path:
            db_file_path = 'image_metadata.db'

        self.dbPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.dbPathLabel.setGeometry(QtCore.QRect(100, 50, 600, 50))
        self.dbPathLabel.setText(f"Database: {db_file_path}")
        self.dbPathLabel.setAlignment(QtCore.Qt.AlignCenter)

        # Display processed image paths
        self.processedImagesTable = QtWidgets.QTableView(self.centralwidget)
        self.processedImagesTable.setGeometry(QtCore.QRect(100, 150, 800, 400))
        self.loadDatabaseData(db_file_path)

        MainWindow.setCentralWidget(self.centralwidget)

    def loadDatabaseData(self, db_file_path):
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()

        cursor.execute("SELECT image_path, predict_path FROM image_metadata")
        rows = cursor.fetchall()

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(['Original Image Path', 'Processed Image Path'])

        for row in rows:
            image_path = row[0]
            processed_path = row[1]
            model.appendRow([QStandardItem(image_path), QStandardItem(processed_path)])

        self.processedImagesTable.setModel(model)
        self.processedImagesTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        conn.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    BasketballCrowdCounting = QtWidgets.QMainWindow()
    ui = Ui_BasketballCrowdCounting()
    ui.setupUi(BasketballCrowdCounting)
    BasketballCrowdCounting.show()
    sys.exit(app.exec_())
