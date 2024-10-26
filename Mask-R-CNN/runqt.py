import os
import cv2
import sqlite3
import time
import torch
import torchvision
from torchvision import transforms as T
from torchvision.ops import nms
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene
from PyQt5.QtGui import QPixmap
from PIL import Image, ImageOps
import numpy as np

class Worker(QtCore.QObject):
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



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1366, 768)
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
        self.worker = Worker()

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



# The following would go in your main app execution logic
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
