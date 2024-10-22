import os
import sqlite3
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QTableView, QHeaderView
from PyQt5.QtGui import QStandardItemModel, QStandardItem
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

        # Credit Button
        self.CreditButton = QtWidgets.QPushButton(self.centralwidget)
        self.CreditButton.setObjectName("CreditButton")
        self.gridLayout.addWidget(self.CreditButton, 9, 0, 1, 1)

        #Open Database Button
        self.OpenDatabase = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(26)
        self.OpenDatabase.setFont(font)
        self.OpenDatabase.setObjectName("OpenDatabase")
        self.gridLayout.addWidget(self.OpenDatabase, 1, 1, 1, 1)

        #Folder Label
        self.FolderLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.FolderLabel.setFont(font)
        self.FolderLabel.setObjectName("FolderLabel")
        self.gridLayout.addWidget(self.FolderLabel, 6, 0, 1, 1)

        #Total Number
        self.Number = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Number.setFont(font)
        self.Number.setObjectName("Number")
        self.gridLayout.addWidget(self.Number, 6, 1, 1, 1)

        #Table to display folder content
        self.tableFolder = QtWidgets.QTableView(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.tableFolder.setFont(font)
        self.tableFolder.setObjectName("tableFolder")
        self.gridLayout.addWidget(self.tableFolder, 4, 0, 1, 1)

        #OpenFolderButton
        self.OpenFolderButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(26)
        self.OpenFolderButton.setFont(font)
        self.OpenFolderButton.setObjectName("OpenFolderButton")
        self.OpenFolderButton.clicked.connect(self.openFolderDialog)  # Connect the button to openFolderDialog
        self.gridLayout.addWidget(self.OpenFolderButton, 1, 0, 1, 1)

        #Start Process Button
        self.ProcessButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.ProcessButton.setFont(font)
        self.ProcessButton.setObjectName("ProcessButton")
        self.ProcessButton.setEnabled(False)  # Initially disable the Process button
        self.ProcessButton.clicked.connect(lambda: self.openNewPage(BasketballCrowdCounting))  # Pass window instance
        self.gridLayout.addWidget(self.ProcessButton, 7, 0, 1, 1)

        #Number Count
        self.CountImg = QtWidgets.QLCDNumber(self.centralwidget)
        self.CountImg.setObjectName("CountImg")
        self.gridLayout.addWidget(self.CountImg, 7, 1, 1, 1)

        BasketballCrowdCounting.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(BasketballCrowdCounting)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 26))
        self.menubar.setObjectName("menubar")
        BasketballCrowdCounting.setMenuBar(self.menubar)

        self.retranslateUi(BasketballCrowdCounting)
        QtCore.QMetaObject.connectSlotsByName(BasketballCrowdCounting)

    def retranslateUi(self, BasketballCrowdCounting):
        _translate = QtCore.QCoreApplication.translate
        BasketballCrowdCounting.setWindowTitle(_translate("BasketballCrowdCounting", "Basketball Court Crowd Counting"))
        self.CreditButton.setText(_translate("BasketballCrowdCounting", "Credit"))
        self.OpenDatabase.setText(_translate("BasketballCrowdCounting", "Database"))
        self.FolderLabel.setText(_translate("BasketballCrowdCounting", "No Folder Selected"))
        self.Number.setText(_translate("BasketballCrowdCounting", "Total Picture"))
        self.OpenFolderButton.setText(_translate("BasketballCrowdCounting", "Open Folder"))
        self.ProcessButton.setText(_translate("BasketballCrowdCounting", "Start Process"))

    def openFolderDialog(self):
        # Open a folder dialog and get the selected folder path
        folder = QFileDialog.getExistingDirectory(None, 'Select Folder')
        if folder:
            self.FolderLabel.setText(f'Selected Folder: {folder}')
            self.selectedFolder = folder
            self.ProcessButton.setEnabled(True)  # Enable the process button
            self.processImages()
            self.showTable(folder)

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

        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time TEXT,
                people_count INTEGER DEFAULT NULL,
                image_path TEXT,
                predict_path TEXT DEFAULT NULL
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

                    # Insert the extracted data into the database, setting people_count and predict_path as None
                    cursor.execute('''
                        INSERT INTO image_metadata (date, time, people_count, image_path, predict_path)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (date, time, None, image_path, None))

                    print(f"Processed: {image_path} - Date: {date}, Time: {time}")

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

        # Commit the transaction and close the connection
        conn.commit()
        conn.close()

        # Update the LCD display and label after processing
        self.CountImg.display(image_count)
        self.FolderLabel.setText(f'Images loaded successfully!')
        self.OpenFolderButton.setEnabled(False) # Diable Open Folder Button
        self.OpenDatabase.setEnabled(False) #Disble Open Database Button
    def showTable(self, folder):
        # Create a model to hold the data
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(['Path'])  # Only one header for the path

        # Loop through all files in the folder and add them to the model
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Only include image files
                image_path = os.path.join(folder, filename)

                # Add row to the model with only the image path
                model.appendRow([QStandardItem(image_path)])

        # Set the model to the table view
        self.tableFolder.setModel(model)
        self.tableFolder.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Make the column stretch to fit


    def openNewPage(self, BasketballCrowdCounting):
        # Switch to the new UI in the same window
        self.new_page_ui = Ui_SecondPage()  # Create a new instance of the second UI
        self.new_page_ui.setupUi(BasketballCrowdCounting)  # Pass the main window (BasketballCrowdCounting)


# Second UI class for the new page
class Ui_SecondPage(object):
    def setupUi(self, BasketballCrowdCounting):
        BasketballCrowdCounting.setObjectName("SecondPage")
        BasketballCrowdCounting.resize(1366, 768)
        self.centralwidget = QtWidgets.QWidget(BasketballCrowdCounting)
        self.centralwidget.setObjectName("centralwidget")

        # Label for New Page
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 100, 600, 400))
        self.label.setText("Welcome to the second page!")
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        BasketballCrowdCounting.setCentralWidget(self.centralwidget)

        self.retranslateUi(BasketballCrowdCounting)
        QtCore.QMetaObject.connectSlotsByName(BasketballCrowdCounting)

    def retranslateUi(self, BasketballCrowdCounting):
        _translate = QtCore.QCoreApplication.translate
        BasketballCrowdCounting.setWindowTitle(_translate("SecondPage", "Second Page"))

# Main entry point of the application
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Create and show the main window
    BasketballCrowdCounting = QtWidgets.QMainWindow()
    ui = Ui_BasketballCrowdCounting()
    ui.setupUi(BasketballCrowdCounting)
    BasketballCrowdCounting.show()
    
    # Execute the application
    sys.exit(app.exec_())