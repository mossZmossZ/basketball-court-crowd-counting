import os
import sqlite3
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QTableView, QHeaderView, QMessageBox
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime

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
        if self.current_model_index < len(self.selected_models):
            model_name = self.selected_models[self.current_model_index]
            if model_name == "YoloV5":
                self.ui_yolov5 = YOLOv5ProcessingPage(self)
                self.ui_yolov5.setupUi(self.main_window)
                self.ui_yolov5.startProcessing()
            elif model_name == "YoloV8":
                self.ui_yolov8 = YOLOv8ProcessingPage(self)
                self.ui_yolov8.setupUi(self.main_window)
                self.ui_yolov8.startProcessing()
            elif model_name == "Mask-R-CNN":
                self.ui_maskrcnn = MaskRCNNProcessingPage(self)
                self.ui_maskrcnn.setupUi(self.main_window)
                self.ui_maskrcnn.startProcessing()
            elif model_name == "Faster-R-CNN":
                self.ui_fasterrcnn = FasterRCNNProcessingPage(self)
                self.ui_fasterrcnn.setupUi(self.main_window)
                self.ui_fasterrcnn.startProcessing()

            self.current_model_index += 1
        else:
            self.showGraphPage(self.main_window)


    def showGraphPage(self, main_window):
        # Call the GraphPage and pass the main window reference
        self.graph_page = GraphPage()  # No argument
        self.graph_page.setupUi(main_window, self.selected_db_file if hasattr(self, 'selected_db_file') else None)

class YOLOv5ProcessingPage:
    def __init__(self,  controller):
        self.controller = controller

    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 100, 600, 400))
        self.label.setText("Processing YOLOv5...")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        MainWindow.setCentralWidget(self.centralwidget)

    def startProcessing(self):
        QtCore.QTimer.singleShot(2000, self.controller.showNextModelProcess)  # Simulate 2 seconds processing

class YOLOv8ProcessingPage:
    def __init__(self, controller):
        self.controller = controller

    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 100, 600, 400))
        self.label.setText("Processing YOLOv8...")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        MainWindow.setCentralWidget(self.centralwidget)

    def startProcessing(self):
        QtCore.QTimer.singleShot(2000, self.controller.showNextModelProcess)  # Simulate 2 seconds processing

class MaskRCNNProcessingPage:
    def __init__(self, controller):
        self.controller = controller

    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 100, 600, 400))
        self.label.setText("Processing Mask-RCNN...")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        MainWindow.setCentralWidget(self.centralwidget)

    def startProcessing(self):
        QtCore.QTimer.singleShot(2000, self.controller.showNextModelProcess)  # Simulate 2 seconds processing

class FasterRCNNProcessingPage:
    def __init__(self, controller):
        self.controller = controller

    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 100, 600, 400))
        self.label.setText("Processing Faster-RCNN...")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        MainWindow.setCentralWidget(self.centralwidget)

    def startProcessing(self):
        QtCore.QTimer.singleShot(2000, self.controller.showNextModelProcess)  # Simulate 2 seconds processing


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
