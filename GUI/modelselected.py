import os
import sqlite3
import sys
import time
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
        self.OpenFolderButton.clicked.connect(self.openFolderDialog)
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
        self.ProcessButton.clicked.connect(self.openNewPage)
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
        folder = QFileDialog.getExistingDirectory(None, 'Select Folder')
        if folder:
            self.FolderLabel.setText(f'Selected Folder: {folder}')
            self.selectedFolder = folder
            self.ProcessButton.setEnabled(True)  # Enable the process button after loading the folder
            self.processImages()
            self.showTable(folder)

    def processImages(self):
        if not hasattr(self, 'selectedFolder'):
            self.FolderLabel.setText('Please select a folder first!')
            return

        folder_path = self.selectedFolder
        conn = sqlite3.connect('image_metadata.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time TEXT,
                people_count INTEGER DEFAULT NULL,
                image_path TEXT,
                predict_path TEXT DEFAULT NULL,
                model TEXT DEFAULT NULL
            )
        ''')

        image_count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                image_count += 1
                try:
                    image = Image.open(image_path)
                    exifdata = image.getexif()
                    date, time = None, None
                    for tagid in exifdata:
                        tagname = TAGS.get(tagid, tagid)
                        value = exifdata.get(tagid)
                        if tagname == 'DateTime':
                            datetime_str = value
                            dt_obj = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
                            date = dt_obj.strftime('%Y-%m-%d')
                            time = dt_obj.strftime('%H:%M')
                    cursor.execute('''
                        INSERT INTO image_metadata (date, time, people_count, image_path, predict_path, model)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (date, time, None, image_path, None, None))
                    print(f"Processed: {image_path} - Date: {date}, Time: {time}")
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

        conn.commit()
        conn.close()

        self.CountImg.display(image_count)
        self.FolderLabel.setText(f'Images loaded successfully!')
        self.OpenFolderButton.setEnabled(False)
        self.OpenDatabase.setEnabled(False)

    def showTable(self, folder):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(['Path'])
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder, filename)
                model.appendRow([QStandardItem(image_path)])
        self.tableFolder.setModel(model)
        self.tableFolder.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def openNewPage(self, BasketballCrowdCounting):
        selected_models = self.get_selected_models()
        if selected_models:
            print(f"Type of BasketballCrowdCounting being passed: {type(BasketballCrowdCounting)}")  # Debug
            self.controller = Controller(BasketballCrowdCounting)  # Pass the actual QMainWindow, not self or False
            self.controller.startProcessing(selected_models)
        else:
            QMessageBox.warning(None, "No Model Selected", "Please select at least one model.")

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

# Implement the Controller class from the previous step here
class Controller(QtWidgets.QMainWindow):
    def __init__(self, main_window):  # main_window should be the actual QMainWindow
        super().__init__()
        self.main_window = main_window  # Store the QMainWindow instance here
        print(f"Passed main_window to Controller: {type(main_window)}")  # Debug
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
                self.ui_yolov5.setupUi(self.main_window)  # Pass the QMainWindow here
                self.ui_yolov5.startProcessing()
            elif model_name == "YoloV8":
                self.ui_yolov8 = YOLOv8ProcessingPage(self)
                self.ui_yolov8.setupUi(self.main_window)  # Pass the QMainWindow here
                self.ui_yolov8.startProcessing()
            elif model_name == "Mask-R-CNN":
                self.ui_maskrcnn = MaskRCNNProcessingPage(self)
                self.ui_maskrcnn.setupUi(self.main_window)  # Pass the QMainWindow here
                self.ui_maskrcnn.startProcessing()
            elif model_name == "Faster-R-CNN":
                self.ui_fasterrcnn = FasterRCNNProcessingPage(self)
                self.ui_fasterrcnn.setupUi(self.main_window)  # Pass the QMainWindow here
                self.ui_fasterrcnn.startProcessing()

            self.current_model_index += 1
        else:
            self.showFinishedPage()

    def showFinishedPage(self):
        self.finished_page = FinishedPage(self)
        self.finished_page.setupUi(self.main_window)


    def showFinishedPage(self):
        self.finished_page = FinishedPage(self)
        self.finished_page.setupUi(self)


class YOLOv5ProcessingPage:
    def __init__(self, controller):
        self.controller = controller

    def setupUi(self, MainWindow):  # MainWindow is the actual QMainWindow
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 100, 600, 400))
        self.label.setText("Processing YOLOv5...")  # Just show a label indicating the process
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
        # Simulate processing
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
        # Simulate processing
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
        # Simulate processing
        QtCore.QTimer.singleShot(2000, self.controller.showNextModelProcess)  # Simulate 2 seconds processing


class FinishedPage:
    def __init__(self, controller):
        self.controller = controller

    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 100, 600, 400))
        self.label.setText("Processing Complete!")
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Button to show the graph page
        self.showGraphButton = QtWidgets.QPushButton("Show Graph", self.centralwidget)
        self.showGraphButton.setGeometry(QtCore.QRect(200, 500, 200, 50))
        self.showGraphButton.clicked.connect(self.showGraphPage)

        MainWindow.setCentralWidget(self.centralwidget)

    def showGraphPage(self):
        self.graph_page = GraphPage(self.controller)
        self.graph_page.setupUi(self.controller)


class GraphPage:
    def __init__(self, controller):
        self.controller = controller

    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 100, 600, 400))
        self.label.setText("Graph Page (Graph will be shown here)")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        MainWindow.setCentralWidget(self.centralwidget)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    BasketballCrowdCounting = QtWidgets.QMainWindow()  # This is the actual QMainWindow
    ui = Ui_BasketballCrowdCounting()
    ui.setupUi(BasketballCrowdCounting)
    
    # Initialize the Controller with the actual QMainWindow
    BasketballCrowdCounting.show()
    sys.exit(app.exec_())
