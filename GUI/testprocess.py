import os
import cv2
import sqlite3
import time
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QApplication, QMainWindow, QVBoxLayout, QWidget, QCalendarWidget, QPushButton, QMessageBox, QLabel, QHBoxLayout, QSizePolicy, QTableWidget, QTableWidgetItem, QComboBox
from PyQt5.QtGui import QPixmap, QColor, QFont
from PyQt5.QtCore import QLocale, Qt, QDate
from ultralytics import YOLO

# คลาสสำหรับการจัดการกับฐานข้อมูล SQLite
class DatabaseManager:
    def __init__(self, db_name='image_metadata.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    # ฟังก์ชันสำหรับสร้างตารางถ้ายังไม่มี
    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time TEXT,
                people_count INTEGER,
                image_path TEXT,
                predict_path TEXT,
                model_name TEXT
            )
        ''')
        self.conn.commit()

    # ฟังก์ชันเพิ่มข้อมูลลงในตาราง
    def insert_record(self, date, time, people_count, image_path):
        self.cursor.execute('''
            INSERT INTO image_metadata (date, time, people_count, image_path, predict_path, model_name)
            VALUES (?, ?, ?, ?, ?)
        ''', (date, time, people_count, image_path))
        self.conn.commit()

    # ฟังก์ชันลบข้อมูลตาม id
    def delete_record(self, record_id):
        self.cursor.execute('''
            DELETE FROM image_metadata WHERE id = ?
        ''', (record_id,))
        self.conn.commit()

    def fetch_all_dates_with_data(self):
        query = "SELECT DISTINCT date FROM image_metadata"
        self.cursor.execute(query)
        return [row[0] for row in self.cursor.fetchall()]
    
    def fetch_Highest_Month_Data(self, selected_date):
        year_month = selected_date.toString("yyyy-MM")  # Get the year and month as a string

        query = f"""
        SELECT date, time, people_count
        FROM (
            SELECT date, time, people_count,
                ROW_NUMBER() OVER (PARTITION BY date ORDER BY people_count DESC) AS rn
            FROM image_metadata
            WHERE strftime('%Y-%m', date) = '{year_month}'  -- Only for the selected month
        ) AS ranked
        WHERE rn = 1
        """

        self.cursor.execute(query)
        return self.cursor.fetchall()

    def fetch_data_by_date(self, date, model_name):
        query = """
        SELECT date, time, people_count
        FROM image_metadata
        WHERE date = ? AND model_name = ?
        ORDER BY people_count DESC
        LIMIT 1
        """
        self.cursor.execute(query, (date, model_name))
        return self.cursor.fetchall()
    
    def fetch_people_count_by_image_path(self, predict_path):
        query = "SELECT people_count FROM image_metadata WHERE predict_path = ?"
        self.cursor.execute(query, (predict_path,))
        result = self.cursor.fetchone()
        return result[0] if result else 0  # Return the count or 0 if no record found

    def fetch_predict_image_paths_by_date(self, date, model_name):
        query = "SELECT predict_path FROM image_metadata WHERE date = ? AND model_name = ?"
        self.cursor.execute(query, (date, model_name))
        results = self.cursor.fetchall()  # Fetch all matching records
        return [result[0] for result in results] if results else [] 
    
    def fetch_predict_image_paths_by_month(self, selected_date):
        year_month = selected_date.toString("yyyy-MM")  # Format date to year-month

        query = f"""
        SELECT predict_path
        FROM image_metadata
        WHERE strftime('%Y-%m', date) = '{year_month}'
        """

        self.cursor.execute(query)
        return self.cursor.fetchall()

    def Line_Chart_fetch_data_by_date(self, date, model_name):
        query = '''
            SELECT time, MAX(people_count) AS total_count
            FROM image_metadata
            WHERE date = ? AND model_name = ?
            GROUP BY time
            ORDER BY total_count DESC
        '''
        self.cursor.execute(query, (date, model_name))
        results = self.cursor.fetchall()  # Get the results as a list of tuples

        # Convert to a dictionary
        return {time: total_count for time, total_count in results}

    # ปิดการเชื่อมต่อฐานข้อมูล
    def close(self):
        self.conn.close()

class GraphPlotter:
    def __init__(self, canvas):
        self.canvas = canvas

    def plot_line_chart_by_day(self, data):
        # Clear any existing axes
        self.canvas.figure.clear()
        
        # Create a new subplot for the line chart
        ax = self.canvas.figure.add_subplot(1, 1, 1)  # Single subplot
        
        categories = list(data.keys())
        values = list(data.values())

        # Line chart
        ax.plot(categories, values, marker='o')
        ax.set_title('People Count Over Day')
        ax.set_xlabel('Time')
        ax.set_ylabel('People Count')
        ax.grid(True)

    def plot_line_chart_by_month(self, data):
        dates = [row[0] for row in data]  # Extract dates
        people_counts = [row[2] for row in data]  # Extract corresponding people counts

        plt.clf()  # Clear the current figure
        plt.plot(dates, people_counts, marker='o')  # Plot line chart
        plt.xlabel('Date')
        plt.ylabel('People Count')
        plt.title('People Count Over Month')

    def draw(self):
        self.canvas.draw()

class CustomCalendarWidget(QCalendarWidget):
    def __init__(self, db_manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_manager = db_manager

    def paintCell(self, painter, rect, date):
        super().paintCell(painter, rect, date)
        formatted_date = date.toString("yyyy-MM-dd")
        if formatted_date in self.db_manager.fetch_all_dates_with_data():
            circle_color = QColor("red")
            circle_radius = 3
            circle_x = int(rect.x() + rect.width() / 2)
            padding = 7
            circle_y = int(rect.y() + rect.height() - padding)
            painter.setBrush(circle_color)
            painter.drawEllipse(circle_x - circle_radius, circle_y - circle_radius, 
                                circle_radius * 2, circle_radius * 2)

class CustomComboBox(QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initial arrow icon
        self.setStyleSheet("""
            QComboBox {
                border: 2px solid #4CAF50;  /* Green border */
                border-radius: 5px;         /* Rounded corners */
                padding: 10px;                /* Inner padding */
                font-size: 16px;             /* Font size */
            }
            QComboBox::drop-down {
                border: 0;                   /* Remove border from dropdown */
                width: 30px;                 /* Width of the dropdown arrow */
                background: transparent;     /* Transparent background */
            }
            QComboBox::down-arrow {
                image: url(arrow_down.png);  /* Path to down arrow icon */
                width: 15px;                 /* Width of the icon */
                height: 15px;                /* Height of the icon */
            }
            QComboBox::up-arrow {
                image: url(arrow_up.png);    /* Path to up arrow icon */
                width: 15px;                 /* Width of the icon */
                height: 15px;                /* Height of the icon */
            }
        """)

    def showPopup(self):
        super().showPopup()
        self.setStyleSheet(self.styleSheet() + " QComboBox::down-arrow { image: url(arrow_up.png); }")

    def hidePopup(self):
        super().hidePopup()
        self.setStyleSheet(self.styleSheet() + " QComboBox::down-arrow { image: url(arrow_down.png); }")


class HoverButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("background-color: #4CAF50; color: white; border: none; border-radius: 5px; padding: 10px;")
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, event):
        self.setStyleSheet("background-color: #45a049; color: white; border: none; border-radius: 5px; padding: 10px;")
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("background-color: #4CAF50; color: white; border: none; border-radius: 5px; padding: 10px;")
        super().leaveEvent(event)

class ShowGraphProcess(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Crown Counting System')
        self.setGeometry(100, 100, 1700, 1000)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Database Manager
        self.db_manager = DatabaseManager()

        # Model Name Combo Box
        self.model_combo_box = CustomComboBox()
        model_names = [
            "YoloV5",
            "YoloV8",
            "Mask-R-CNN",
            "Faster-R-CNN",
            "CSRNet"
        ]
        self.model_combo_box.addItems(model_names)

        # Set the size for the combo box
        self.model_combo_box.setFixedSize(800, 40)

        # Create a layout for the combo box and add it to the main layout
        combo_layout = QHBoxLayout()
        combo_layout.setContentsMargins(10, 10, 10, 10)
        combo_layout.addWidget(self.model_combo_box)
        layout.addLayout(combo_layout)

        # Custom Calendar Widget
        self.calendar = CustomCalendarWidget(self.db_manager, self)
        self.calendar.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.calendar.setGridVisible(True)
        self.calendar.setFixedHeight(350)

        layout.addWidget(self.calendar)

        # Button to show graphs
        self.plot_button = HoverButton('Show Graphs / Re-Graphs')
        self.plot_button.clicked.connect(self.show_graph_by_seleted_day)
        self.plot_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.plot_button)

        # Horizontal layout for image, graph, and labels
        self.labels_and_canvas_layout = QHBoxLayout()

        # Image display widget
        self.image_widget = QWidget()
        image_layout = QVBoxLayout(self.image_widget)

        # New label for number of people above the image
        self.people_count_label = QLabel("People Count")
        self.people_count_label.setAlignment(Qt.AlignCenter)
        self.people_count_label.setFont(QFont('Arial', 16))

        image_layout.addWidget(self.people_count_label)  # Add the count label above the image

        self.image_label = QLabel()
        self.image_label.setScaledContents(True)  # Scale image to fit the label
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.image_label)

        # Navigation buttons
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_image)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_image)

        # Add buttons to the image layout
        image_layout.addWidget(self.prev_button)
        image_layout.addWidget(self.next_button)

        self.image_widget.setFixedSize(400, 500)  # Set a fixed size for the image widget
        self.labels_and_canvas_layout.addWidget(self.image_widget)  # Add image widget to layout

        # Matplotlib figure and canvas
        self.figure = plt.figure(figsize=(10, 4))  # Increase width, keep height the same
        self.canvas = FigureCanvas(self.figure)

        # Graph frame
        self.graph_frame = QWidget()
        graph_layout = QVBoxLayout(self.graph_frame)
        graph_layout.addWidget(self.canvas)
        self.graph_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph_frame.setMinimumSize(400, 400)  # Increase width, keep height the same
        self.labels_and_canvas_layout.addWidget(self.graph_frame)

        # Vertical layout for labels
        self.labels_layout = QVBoxLayout()
        self.labels_layout.setContentsMargins(0, 0, 20, 0)

        # Number of people label
        self.number_frame = QWidget()
        self.number_frame.setStyleSheet("background-color: white; padding: 5px")

        number_layout = QVBoxLayout(self.number_frame)
        number_layout.setContentsMargins(5, 5, 10, 5)

        # Title for Number of People
        self.number_of_people_title = QLabel("Number of People")
        self.number_of_people_title.setAlignment(Qt.AlignCenter)
        self.number_of_people_title.setFont(QFont('Arial', 10))
        number_layout.addWidget(self.number_of_people_title)

        # Label for showing the actual number of people
        self.number_of_people_value = QLabel("0")  # Initialize with 0
        self.number_of_people_value.setAlignment(Qt.AlignCenter)
        self.number_of_people_value.setFont(QFont('Arial', 30, QFont.Bold))
        self.number_of_people_value.setStyleSheet("color: #4CAF50")
        number_layout.addWidget(self.number_of_people_value)

        self.number_frame.setFixedSize(420, 150)
        self.labels_layout.addWidget(self.number_frame)

        # Create QTableWidget for displaying records
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(2)
        self.data_table.setHorizontalHeaderLabels(["Date (Highest each Day)", "People"])
        self.data_table.setFixedSize(420, 353)
        self.labels_layout.addWidget(self.data_table)

        # Add labels layout to the horizontal layout
        self.labels_and_canvas_layout.addLayout(self.labels_layout)

        # Add the labels and canvas layout to the main layout
        layout.addLayout(self.labels_and_canvas_layout)

        current_date = QDate.currentDate()
        self.calendar.setCurrentPage(current_date.year(), current_date.month())  # Set the calendar to the current month
        self.calendar.setSelectedDate(current_date)  # Optionally select today's date

        self.central_widget.setLayout(layout)
        self.graph_plotter = GraphPlotter(self.canvas)

        # Load data for the current month
        self.update_data_based_on_month_change(current_date.year(), current_date.month())
        
        self.calendar.currentPageChanged.connect(self.update_data_based_on_month_change)


    def update_data_based_on_month_change(self, year, month):
        # Create a QDate object for the first day of the new month
        selected_date = QDate(year, month, 1)

        # Fetch and show the graph for the new month
        self.show_graph_by_month(selected_date)

        # Fetch records for the highest people count
        records = self.db_manager.fetch_Highest_Month_Data(selected_date)
        self.populate_table(records, 1)

        # Clear the image paths and reset the current image index
        self.image_paths = []  # Clear the image paths
        self.current_image_index = 0  # Reset index

        # Fetch images for the selected month
        image_paths = self.db_manager.fetch_predict_image_paths_by_month(selected_date)

        # Update the image_paths with the new data
        self.image_paths = [path[0] for path in image_paths]  # Unwrap tuples

        # Load the first image if available
        self.load_image()  # Load the first image of the new month

    def load_image(self):
        if not self.image_paths:
            print("No images available to load.")
            return  # Exit early if there are no images

        if self.current_image_index >= len(self.image_paths):
            print(f"Index out of range: {self.current_image_index} >= {len(self.image_paths)}")
            self.current_image_index = len(self.image_paths) - 1  # Reset to last index
            return  # Exit early since index is out of bounds

        print("Loading Image...")
        pixmap = QPixmap(self.image_paths[self.current_image_index])  # Load the current image
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

        current_people_count = self.db_manager.fetch_people_count_by_image_path(self.image_paths[self.current_image_index])
        self.people_count_label.setText(f"People Count : {current_people_count}")

        print("Total Images:", len(self.image_paths))
        self.prev_button.setVisible(self.current_image_index > 0)
        self.next_button.setVisible(self.current_image_index < len(self.image_paths) - 1)


    def show_previous_image(self):
        if self.image_paths and self.current_image_index > 0:  # Check that we're not at the first image
            self.current_image_index -= 1  # Decrement index
            self.load_image()  # Load the previous image

    def show_next_image(self):
        if self.image_paths and self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_image()

    def format_date(self, date_str, time_str):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%B %d %Y at")
        return f"{formatted_date} {time_str}"
    
    def show_graph_by_month(self, selected_date):
        monthly_data = self.db_manager.fetch_Highest_Month_Data(selected_date)  # Pass the QDate object
        if monthly_data:
            self.graph_plotter.plot_line_chart_by_month(monthly_data)  # Call your plotting function
            self.graph_plotter.draw()  # Refresh the canvas
        else:
            year_month = selected_date.toString("yyyy-MM")
            QMessageBox.warning(self, "No Data", "No data available for the " + year_month, QMessageBox.Ok)

    def show_graph_by_seleted_day(self):
        selected_date = self.calendar.selectedDate().toString("yyyy-MM-dd")
        selected_model = self.model_combo_box.currentText()

        # Fetch line chart data for the selected date
        line_chart_data = self.db_manager.Line_Chart_fetch_data_by_date(selected_date, selected_model)

        # Fetch all image paths for the selected date
        self.image_paths = self.db_manager.fetch_predict_image_paths_by_date(selected_date, selected_model)

        if line_chart_data:
            self.graph_plotter.plot_line_chart_by_day(line_chart_data)
            self.graph_plotter.draw()

            # Populate the table with data
            records = self.db_manager.fetch_data_by_date(selected_date, selected_model)
            self.populate_table(records, 0)

            self.data_table.setHorizontalHeaderLabels(["Date (Peak Time)", "People"])

            # Load the first image
            self.load_image()
        else:
            QMessageBox.warning(self, "Caution", f"No data available for the model '{selected_model}' on date: {selected_date}.", QMessageBox.Ok)

        self.current_image_index = 0  # Reset to the first image
            
    def populate_table(self, records, status):
        if status == 0:
            self.data_table.setRowCount(len(records))  # Set the number of rows based on records
            for row_index, record in enumerate(records):
                date = record[0]
                time = record[1]  
                people_count = record[2] 

                formatted_date = self.format_date(date, time)
                self.data_table.setItem(row_index, 0, QTableWidgetItem(formatted_date))  # Set date in column 0

                self.number_of_people_title.setText("Number of People (Selected Day)")
                self.number_of_people_value.setText(str(people_count))


                # Set the people count
                people_item = QTableWidgetItem(str(people_count))  # Create the item for people count
                people_item.setTextAlignment(Qt.AlignCenter)  # Center-align the text
                self.data_table.setItem(row_index, 1, people_item)  # Set the centered item in column 1

            # Set specific widths for columns
            self.data_table.setColumnWidth(0, 260) 
        elif status == 1:
            self.data_table.setRowCount(len(records))  # Set the number of rows based on records
            total_people_count = 0  # Initialize total count
            for row_index, record in enumerate(records):
                date = record[0]
                time = record[1]  
                people_count = record[2] 

                formatted_date = self.format_date(date, time)
                self.data_table.setItem(row_index, 0, QTableWidgetItem(formatted_date))  # Set date in column 0

                # Set the people count
                people_item = QTableWidgetItem(str(people_count))  # Create the item for people count
                people_item.setTextAlignment(Qt.AlignCenter)  # Center-align the text
                self.data_table.setItem(row_index, 1, people_item)  # Set the centered item in column 1

                total_people_count += people_count  # Accumulate the total people count

            # Update the number of people label
            
            self.number_of_people_title.setText("Number of People (In Month)")
            self.number_of_people_value.setText(str(total_people_count))

            # Set specific widths for columns
            self.data_table.setColumnWidth(0, 260)

    def closeEvent(self, event):
        self.db_manager.close()
        event.accept()

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
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.People_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.People_label.setFont(font)
        self.People_label.setObjectName("People_label")
        self.verticalLayout.addWidget(self.People_label)
        self.lcd_count = QtWidgets.QLCDNumber(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.lcd_count.setFont(font)
        self.lcd_count.setObjectName("lcd_count")
        self.verticalLayout.addWidget(self.lcd_count)
        self.gridLayout.addLayout(self.verticalLayout, 0, 2, 2, 1)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 1)
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout.addWidget(self.graphicsView_2, 1, 1, 1, 1)
        self.Remaining_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.Remaining_label.setFont(font)
        self.Remaining_label.setObjectName("Remaining_label")
        self.gridLayout.addWidget(self.Remaining_label, 2, 0, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.progressBar.setFont(font)
        self.progressBar.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 2, 1, 1, 1)
        self.process_text = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.process_text.setFont(font)
        self.process_text.setObjectName("process_text")
        self.gridLayout.addWidget(self.process_text, 3, 0, 1, 1)
        self.ProcessImageButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.ProcessImageButton.setFont(font)
        self.ProcessImageButton.setObjectName("ProcessImageButton")
        self.gridLayout.addWidget(self.ProcessImageButton, 3, 1, 1, 1)

        self.GraphButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.GraphButton.setFont(font)
        self.GraphButton.setObjectName("GraphButton")
        self.GraphButton.setText("Show Graph Data")
        self.gridLayout.addWidget(self.GraphButton, 4, 1, 1, 1)
        self.GraphButton.hide()  # ซ่อนปุ่มนี้ไว้ก่อน

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
        self.ProcessImageButton.clicked.connect(self.processImages)
        self.GraphButton.clicked.connect(self.run_app)

    def run_app(self):
        self.graph_window = ShowGraphProcess()  # Create a new instance of the graph window
        self.graph_window.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Remaining_label.setText(_translate("MainWindow", "Remaining"))
        self.ProcessImageButton.setText(_translate("MainWindow", "Process Image"))
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

        self.GraphButton.show()
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
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
