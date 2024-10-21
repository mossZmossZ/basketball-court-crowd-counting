import os
import sqlite3
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime

class ImageMetadataApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the GUI layout
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Metadata Extractor')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        # Label to show the selected folder
        self.folderLabel = QLabel('No folder selected', self)
        layout.addWidget(self.folderLabel)

        # Button to open folder dialog
        self.folderButton = QPushButton('Select Folder', self)
        self.folderButton.clicked.connect(self.openFolderDialog)
        layout.addWidget(self.folderButton)

        # Button to start processing
        self.processButton = QPushButton('Process Images', self)
        self.processButton.clicked.connect(self.processImages)
        layout.addWidget(self.processButton)

        self.setLayout(layout)

    def openFolderDialog(self):
        # Open a folder dialog and get the selected folder path
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder:
            self.folderLabel.setText(f'Selected Folder: {folder}')
            self.selectedFolder = folder

    def processImages(self):
        # If the button label is "Exit", close the application
        if self.processButton.text() == 'Exit':
            QApplication.quit()  # Close the application
            return

        # Ensure a folder is selected
        if not hasattr(self, 'selectedFolder'):
            self.folderLabel.setText('Please select a folder first!')
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
                path TEXT
            )
        ''')

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Filter image files
                image_path = os.path.join(folder_path, filename)

                try:
                    # open the image
                    image = Image.open(image_path)

                    # extracting the exif metadata
                    exifdata = image.getexif()

                    # Initialize variables for date and time
                    date = None
                    time = None

                    # Loop through all the tags present in exifdata
                    for tagid in exifdata:
                         
                        # getting the tag name instead of tag id
                        tagname = TAGS.get(tagid, tagid)
                     
                        # passing the tagid to get its respective value
                        value = exifdata.get(tagid)

                        # looking for the DateTime tag
                        if tagname == 'DateTime':
                            # original format: 'YYYY:MM:DD HH:MM:SS'
                            datetime_str = value
                            
                            # converting to datetime object
                            dt_obj = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
                            
                            # formatting the date as YYYY-MM-DD
                            date = dt_obj.strftime('%Y-%m-%d')
                            
                            # formatting the time as HH:MM
                            time = dt_obj.strftime('%H:%M')

                    # Insert the extracted data into the database
                    cursor.execute('''
                        INSERT INTO image_metadata (date, time, path)
                        VALUES (?, ?, ?)
                    ''', (date, time, image_path))

                    print(f"Processed: {image_path} - Date: {date}, Time: {time}")

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

        # Commit the transaction and close the connection
        conn.commit()
        conn.close()

        # Update label and button after processing
        self.folderLabel.setText(f'Images from {folder_path} processed successfully!')
        self.processButton.setText('Exit')  # Change the button text to "Exit"

# Run the PyQt5 application
if __name__ == '__main__':
    app = QApplication([])
    window = ImageMetadataApp()
    window.show()
    app.exec_()
