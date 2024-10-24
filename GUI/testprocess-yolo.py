import os
import cv2
import sqlite3
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene
from PyQt5.QtGui import QPixmap
from ultralytics import YOLO

# Create a Worker class to handle background image processing
class Worker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, str, str, int, str)  # Signal to update progress (progress, original image path, predicted image path, people_count, remaining_time)
    completed = QtCore.pyqtSignal(str)                     # Signal for completion message

    def __init__(self):
        super().__init__()
        self.model = YOLO('yolov8x.pt')
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
        output_folder = os.path.join(os.getcwd(), 'PredictIMG/yolo')
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
            ''', (people_count, predict_image_path, 'YOLOv8', image_id))
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
        font.setPointSize(18)
        self.Original_label.setFont(font)
        self.Original_label.setObjectName("Original_label")
        self.gridLayout.addWidget(self.Original_label, 0, 0, 1, 1)
        self.Predict_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.Predict_label.setFont(font)
        self.Predict_label.setObjectName("Predict_label")
        self.gridLayout.addWidget(self.Predict_label, 0, 1, 1, 1)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 1)
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout.addWidget(self.graphicsView_2, 1, 1, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout.addLayout(self.verticalLayout, 0, 2, 2, 1)
        self.Remaining_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.Remaining_label.setFont(font)
        self.Remaining_label.setObjectName("Remaining_label")
        self.gridLayout.addWidget(self.Remaining_label, 4, 0, 1, 1)
        self.ShowGraphButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.ShowGraphButton.setFont(font)
        self.ShowGraphButton.setObjectName("ShowGraphButton")
        self.gridLayout.addWidget(self.ShowGraphButton, 7, 1, 1, 1)
        self.People_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.People_label.setFont(font)
        self.People_label.setObjectName("People_label")
        self.gridLayout.addWidget(self.People_label, 2, 0, 1, 1)
        self.lcd_count = QtWidgets.QLCDNumber(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.lcd_count.setFont(font)
        self.lcd_count.setObjectName("lcd_count")
        self.gridLayout.addWidget(self.lcd_count, 2, 1, 1, 1)
        self.process_text = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.process_text.setFont(font)
        self.process_text.setObjectName("process_text")
        self.gridLayout.addWidget(self.process_text, 4, 1, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.progressBar.setFont(font)
        self.progressBar.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 6, 0, 1, 2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 7, 0, 1, 1)
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
        self.ShowGraphButton.clicked.connect(self.startProcessing)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Original_label.setText(_translate("MainWindow", "Original Image"))
        self.Predict_label.setText(_translate("MainWindow", "Predict Image"))
        self.Remaining_label.setText(_translate("MainWindow", "Time Remaining :"))
        self.ShowGraphButton.setText(_translate("MainWindow", "Show Graph"))
        self.People_label.setText(_translate("MainWindow", "People Count :"))
        self.process_text.setText(_translate("MainWindow", "Processing :"))
        self.pushButton.setText(_translate("MainWindow", "Start Process"))

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
        scene = QGraphicsScene()
        pixmap = QPixmap(img_path)
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
