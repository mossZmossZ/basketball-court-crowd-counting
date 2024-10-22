import os
import cv2
import sqlite3
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene
from PyQt5.QtGui import QPixmap
from ultralytics import YOLO

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1366, 768)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        #Graphics 1
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout.addWidget(self.graphicsView_2, 3, 1, 1, 1)

        #Graphics 2
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 3, 0, 1, 1)

        #People_label
        self.People_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.People_label.setFont(font)
        self.People_label.setObjectName("People_label")
        self.gridLayout.addWidget(self.People_label, 0, 2, 1, 1)

        #ShowGraphButton
        self.ShowGraphButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.ShowGraphButton.setFont(font)
        self.ShowGraphButton.setObjectName("ShowGraphButton")
        self.gridLayout.addWidget(self.ShowGraphButton, 5, 1, 1, 1)

        #Original Image label
        self.Original_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(18)
        self.Original_label.setFont(font)
        self.Original_label.setObjectName("Original_label")
        self.gridLayout.addWidget(self.Original_label, 0, 0, 1, 1)

        #process text label
        self.process_text = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.process_text.setFont(font)
        self.process_text.setObjectName("process_text")
        self.gridLayout.addWidget(self.process_text, 5, 0, 1, 1)

        #progressBar
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.progressBar.setFont(font)
        self.progressBar.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 4, 1, 1, 1)

        #Predict IMG label
        self.Predict_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.Predict_label.setFont(font)
        self.Predict_label.setObjectName("Predict_label")
        self.gridLayout.addWidget(self.Predict_label, 0, 1, 1, 1)

        #Remaining label
        self.Remaining_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.Remaining_label.setFont(font)
        self.Remaining_label.setObjectName("Remaining_label")
        self.gridLayout.addWidget(self.Remaining_label, 4, 0, 1, 1)

        #lcd count label
        self.lcd_count = QtWidgets.QLCDNumber(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.lcd_count.setFont(font)
        self.lcd_count.setObjectName("lcd_count")
        self.gridLayout.addWidget(self.lcd_count, 3, 2, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1366, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Set up the model for YOLOv8
        self.model = YOLO('yolov8x.pt')
        
        # Connect the button for starting the process
        self.ShowGraphButton.clicked.connect(self.processImages)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Remaining_label.setText(_translate("MainWindow", "Remaining"))
        self.ShowGraphButton.setText(_translate("MainWindow", "Show Graph"))
        self.People_label.setText(_translate("MainWindow", "People"))
        self.Original_label.setText(_translate("MainWindow", "Original Image"))
        self.Predict_label.setText(_translate("MainWindow", "Predict Image"))
        self.process_text.setText(_translate("MainWindow", "Process"))

    def processImages(self):
        # Connect to the SQLite database
        conn = sqlite3.connect('image_metadata.db')
        cursor = conn.cursor()

        # Select all images from the database
        cursor.execute("SELECT id, image_path FROM image_metadata WHERE predict_path IS NULL")
        images = cursor.fetchall()

        total_images = len(images)
        if total_images == 0:
            self.process_text.setText("No images to process.")
            return

        # Create output folder if not exists
        output_folder = os.path.join(os.getcwd(), 'PredictIMG/yolo')
        os.makedirs(output_folder, exist_ok=True)

        # Variables to track time
        start_time = time.time()  # Start time of the entire process
        times_per_image = []

        # Loop through each image in the database
        for index, (image_id, image_path) in enumerate(images):
            # Start timer for processing one image
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
                        # Increase the thickness of the bounding box
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Adjust thickness here (4)

            # Save the predicted image
            predict_image_path = os.path.join(output_folder, f"predict_{os.path.basename(image_path)}")
            cv2.imwrite(predict_image_path, image)

            # Update the database with the prediction results
            cursor.execute('''
                UPDATE image_metadata
                SET people_count = ?, predict_path = ?
                WHERE id = ?
            ''', (people_count, predict_image_path, image_id))
            conn.commit()

            # Stop timer for this image and calculate the time taken
            image_end_time = time.time()
            time_for_image = image_end_time - image_start_time
            times_per_image.append(time_for_image)  # Store the time for this image

            # Estimate remaining time
            avg_time_per_image = sum(times_per_image) / len(times_per_image)
            remaining_time = avg_time_per_image * (total_images - (index + 1))

            # Convert remaining time to min:ss format
            remaining_min, remaining_sec = divmod(int(remaining_time), 60)
            remaining_time_str = f"{remaining_min}:{remaining_sec:02d}"

            # Update GUI with time and progress
            self.lcd_count.display(people_count)  # Show count for the current image
            self.progressBar.setValue(int((index + 1) / total_images * 100))
            self.process_text.setText(f"Processing {index + 1} of {total_images}...")

            # Display remaining time in min:ss format
            self.Remaining_label.setText(f"Time remaining: {remaining_time_str} min")

            # Display original and predicted images in the GUI
            self.displayImage(image_path, self.graphicsView)
            self.displayImage(predict_image_path, self.graphicsView_2)

        # Total time taken for the process
        total_time = time.time() - start_time

        # Convert total time to min:ss format
        total_min, total_sec = divmod(int(total_time), 60)
        total_time_str = f"{total_min}:{total_sec:02d}"

        self.process_text.setText(f"Process completed in {total_time_str} min")
        conn.close()


    def displayImage(self, img_path, graphicsView):
        scene = QGraphicsScene()
        pixmap = QPixmap(img_path)
        
        # Set the image to fill the entire view while maintaining aspect ratio
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        graphicsView.setScene(scene)
        
        # Ensures that the image fills the QGraphicsView area without leaving empty space
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
