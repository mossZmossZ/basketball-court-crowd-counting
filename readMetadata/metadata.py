import os
import sqlite3
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime

# Path to the folder containing the images
folder_path = "../IMG"

# Database setup
conn = sqlite3.connect('image_metadata.db')  # Connect to SQLite database (creates it if it doesn't exist)
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

print("All data saved to the database.")
