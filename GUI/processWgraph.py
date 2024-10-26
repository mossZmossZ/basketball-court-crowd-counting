import os
import cv2
import sqlite3
import time
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QCheckBox, QGraphicsPixmapItem, QGraphicsScene, QApplication, QMainWindow, QVBoxLayout, QWidget, QCalendarWidget, QPushButton, QMessageBox, QLabel, QHBoxLayout, QSizePolicy, QTableWidget, QTableWidgetItem, QComboBox
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
                model TEXT
            )
        ''')
        self.conn.commit()

    # ฟังก์ชันเพิ่มข้อมูลลงในตาราง
    def insert_record(self, date, time, people_count, image_path):
        self.cursor.execute('''
            INSERT INTO image_metadata (date, time, people_count, image_path, predict_path, model)
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
    
    def fetch_Highest_Month_Data(self, selected_date, selected_model):
        year_month = selected_date.toString("yyyy-MM")  # Get the year and month as a string

        query = f"""
        SELECT date, time, people_count
        FROM (
            SELECT date, time, people_count,
                ROW_NUMBER() OVER (PARTITION BY date ORDER BY people_count DESC) AS rn
            FROM image_metadata
            WHERE strftime('%Y-%m', date) = '{year_month}' AND model = '{selected_model}'
        ) AS ranked
        WHERE rn = 1
        """

        self.cursor.execute(query)
        return self.cursor.fetchall()

    def fetch_data_by_date(self, date, model):
        query = """
        SELECT date, time, people_count
        FROM image_metadata
        WHERE date = ? AND model = ?
        ORDER BY people_count DESC
        LIMIT 1
        """
        self.cursor.execute(query, (date, model))
        return self.cursor.fetchall()
    
    def fetch_people_count_by_image_path(self, predict_path):
        query = "SELECT people_count FROM image_metadata WHERE predict_path = ?"
        self.cursor.execute(query, (predict_path,))
        result = self.cursor.fetchone()
        return result[0] if result else 0  # Return the count or 0 if no record found

    def fetch_predict_image_paths_by_date(self, date, model):
        query = "SELECT predict_path FROM image_metadata WHERE date = ? AND model = ?"
        self.cursor.execute(query, (date, model))
        results = self.cursor.fetchall()  # Fetch all matching records
        return [result[0] for result in results] if results else [] 
    
    def fetch_predict_image_paths_by_month(self, selected_date, selected_model):
        year_month = selected_date.toString("yyyy-MM")  # Format date to year-month

        query = f"""
        SELECT predict_path
        FROM image_metadata
        WHERE strftime('%Y-%m', date) = '{year_month}' AND model = '{selected_model}'
        """

        self.cursor.execute(query)
        return self.cursor.fetchall()

    def Line_Chart_fetch_data_by_date(self, date, model):
        query = '''
            SELECT time, MAX(people_count) AS total_count
            FROM image_metadata
            WHERE date = ? AND model = ?
            GROUP BY time
            ORDER BY total_count DESC
        '''
        self.cursor.execute(query, (date, model))
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

class ShowGraphProcess(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Crown Counting System')
        self.setGeometry(100, 100, 1366, 768)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Database Manager
        self.db_manager = DatabaseManager()
        self.image_paths = []  # Initialize to an empty list
        self.current_image_index = 0

        # Create a horizontal layout for the combo box and checkboxes
        combo_layout = QHBoxLayout()
        combo_layout.setSpacing(20)  # Space between items in this layout

        # Model Name Combo Box
        self.model_combo_box = CustomComboBox()
        model = [
            "Selected Model",
            "YoloV5",
            "YoloV8",
            "Mask-R-CNN",
            "Faster-R-CNN",
            "CSRNet"
        ]
        self.model_combo_box.addItems(model)
        self.model_combo_box.setFixedSize(1100, 40)

        self.day_checkbox = QCheckBox("Day (Note : Default is Current Day)")
        self.month_checkbox = QCheckBox("Month")

        combo_layout.addWidget(self.model_combo_box)
        combo_layout.addWidget(self.day_checkbox)
        combo_layout.addWidget(self.month_checkbox)

        # Add the combo layout to the main layout
        layout.addLayout(combo_layout)

        # Custom Calendar Widget
        self.calendar = CustomCalendarWidget(self.db_manager, self)
        self.calendar.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.calendar.setGridVisible(True)
        self.calendar.setFixedHeight(350)

        layout.addWidget(self.calendar)
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

        self.image_widget.setFixedSize(370, 500)  # Set a fixed size for the image widget
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
        self.data_table.setHorizontalHeaderLabels(["Date", "People"])
        self.data_table.setFixedSize(420, 353)
        self.labels_layout.addWidget(self.data_table)

        # Add labels layout to the horizontal layout
        self.labels_and_canvas_layout.addLayout(self.labels_layout)

        # Add the labels and canvas layout to the main layout
        layout.addLayout(self.labels_and_canvas_layout)

        current_date = QDate.currentDate()
        self.calendar.setCurrentPage(current_date.year(), current_date.month())  # Set the calendar to the current month

        self.central_widget.setLayout(layout)
        self.graph_plotter = GraphPlotter(self.canvas)

        self.day_checkbox.toggled.connect(self.on_checkbox_toggled)
        self.month_checkbox.toggled.connect(self.on_checkbox_toggled)
        self.model_combo_box.currentIndexChanged.connect(self.on_combobox_changed)
        self.calendar.selectionChanged.connect(self.show_graph_by_seleted_day) 
        self.calendar.currentPageChanged.connect(self.show_graph_by_seleted_month)

    def load_image(self):
        if self.model_combo_box.currentText() != "Selected Model":
            if not self.image_paths:
                print("No images available to load.")
                return  # Exit early if there are no images

            if self.current_image_index >= len(self.image_paths):
                print(f"Index out of range: {self.current_image_index} >= {len(self.image_paths)}")
                self.current_image_index = len(self.image_paths) - 1  # Reset to last index
                return  # Exit early since index is out of bounds

            pixmap = QPixmap(self.image_paths[self.current_image_index])  # Load the current image
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

            current_people_count = self.db_manager.fetch_people_count_by_image_path(self.image_paths[self.current_image_index])
            self.people_count_label.setText(f"People Count : {current_people_count}")

            self.prev_button.setVisible(self.current_image_index > 0)
            self.next_button.setVisible(self.current_image_index < len(self.image_paths) - 1)

    def show_previous_image(self):
        if self.model_combo_box.currentText() != "Selected Model":
            if self.image_paths and self.current_image_index > 0:  # Check that we're not at the first image
                self.current_image_index -= 1  # Decrement index
                self.load_image()  # Load the previous image

    def show_next_image(self):
        if self.model_combo_box.currentText() != "Selected Model":
            if self.image_paths and self.current_image_index < len(self.image_paths) - 1:
                self.current_image_index += 1
                self.load_image()

    def format_date(self, date_str, time_str):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%B %d %Y at")
        return f"{formatted_date} {time_str}"
    
    def on_calendar_page_change(self):
        current_month = self.calendar.selectedDate().month()
        current_year = self.calendar.selectedDate().year()
        print(current_month)
        self.show_graph_by_seleted_month(current_year, current_month)
    
    def on_checkbox_toggled(self):
        if self.sender() is self.day_checkbox and self.day_checkbox.isChecked():
            self.month_checkbox.setChecked(False)
            self.show_graph_by_seleted_day()
        elif self.sender() is self.month_checkbox and self.month_checkbox.isChecked():
            self.day_checkbox.setChecked(False)
            current_month = self.calendar.selectedDate().month()
            current_year = self.calendar.selectedDate().year()
            self.show_graph_by_seleted_month(current_year, current_month)

    def on_combobox_changed(self):
        if self.month_checkbox.isChecked():
            current_month = self.calendar.selectedDate().month()
            current_year = self.calendar.selectedDate().year()
            self.show_graph_by_seleted_month(current_year, current_month)
        elif self.day_checkbox.isChecked():
            self.show_graph_by_seleted_day()
    
    def show_graph_by_seleted_month(self, year, month):
        selected_date = QDate(year, month, 1)
        print(year)
        print(month)
        selected_model = self.model_combo_box.currentText()
        if selected_model != "Selected Model" and self.month_checkbox.isChecked():
            monthly_data = self.db_manager.fetch_Highest_Month_Data(selected_date, selected_model)  # Pass the QDate object
            if monthly_data:
                self.next_button.setEnabled(True)
                self.prev_button.setEnabled(True)
                self.graph_plotter.plot_line_chart_by_month(monthly_data)  # Call your plotting function
                self.graph_plotter.draw()

                # Clear the image paths and reset the current image index
                self.image_paths = []  # Clear the image paths
                self.current_image_index = 0  # Reset index

                # Fetch images for the selected month
                image_paths = self.db_manager.fetch_predict_image_paths_by_month(selected_date, selected_model)

                # Update the image_paths with the new data
                self.image_paths = [path[0] for path in image_paths]  # Unwrap tuples

                # Load the first image if available
                self.load_image()  # Load the first image of the new month
                # Fetch records for the highest people count
                records = self.db_manager.fetch_Highest_Month_Data(selected_date, selected_model)
                self.data_table.setHorizontalHeaderLabels(["Date (Highest each day)", "People"])
                self.populate_table(records, 1)
            else:
                self.next_button.setEnabled(False)
                self.prev_button.setEnabled(False)
                self.graph_plotter.plot_line_chart_by_month([])
                self.graph_plotter.draw()
                self.people_count_label.setText(f"People Count : 0")
                pixmap = QPixmap("default_image.png")
                self.image_label.setPixmap(pixmap)
                year_month = selected_date.toString("yyyy-MM")
                self.number_of_people_value.setText(str(0))
                self.populate_table([],1)
                QMessageBox.warning(self, "No Data", "No data for model " + selected_model+ " available for the " + year_month, QMessageBox.Ok)

    def show_graph_by_seleted_day(self):
        selected_date = self.calendar.selectedDate().toString("yyyy-MM-dd")
        selected_model = self.model_combo_box.currentText()

        if selected_model != "Selected Model" and self.day_checkbox.isChecked():
            # Fetch line chart data for the selected date
            line_chart_data = self.db_manager.Line_Chart_fetch_data_by_date(selected_date, selected_model)

            # Fetch all image paths for the selected date
            self.image_paths = self.db_manager.fetch_predict_image_paths_by_date(selected_date, selected_model)

            if line_chart_data:
                self.next_button.setEnabled(True)
                self.prev_button.setEnabled(True)
                self.graph_plotter.plot_line_chart_by_day(line_chart_data)
                self.graph_plotter.draw()

                # Populate the table with data
                records = self.db_manager.fetch_data_by_date(selected_date, selected_model)
                self.populate_table(records, 0)

                self.data_table.setHorizontalHeaderLabels(["Date (Peak Time)", "People"])

                # Load the first image
                self.load_image()
            else:
                self.next_button.setEnabled(False)
                self.prev_button.setEnabled(False)
                self.graph_plotter.plot_line_chart_by_month([])
                self.graph_plotter.draw()
                self.people_count_label.setText(f"People Count : 0")
                pixmap = QPixmap("default_image.png")
                self.image_label.setPixmap(pixmap)
                self.number_of_people_value.setText(str(0))
                self.populate_table([],0)
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

# The following would go in your main app execution logic
if __name__ == "__main__":
    # import sqlite3

    # db_name = 'image_metadata.db'
    # conn = sqlite3.connect(db_name)
    # cursor = conn.cursor()

    # cursor.execute("UPDATE image_metadata SET model = 'Mask-R-CNN' WHERE date = '2024-09-16';")
   

    # conn.commit()
    # conn.close()
    app = QApplication(sys.argv)
    window = ShowGraphProcess()
    window.show()
    sys.exit(app.exec_())


    