import os
import pandas as pd
from flask import request
from werkzeug.utils import secure_filename

from modules.file_utils import allowed_file


def handle_uploaded_data():
    # variables for the parameter
    error_msg = None
    success_msg = None
    data_table = None

    dataset_size = 0

    new_filename = "tweets_dataset.csv"
    save_location = os.path.join("database", new_filename)

    columns = ["Username", "Handle", "Timestamp", "Tweet", "Label"]

    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            error_msg = "No file selected."
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            try:
                file.save(save_location)

                success_msg = "File uploaded successfully."

            except Exception as e:
                error_msg = "An error occurred while saving the file."
        else:
            error_msg = "Invalid file. Only CSV files are allowed."

    # show the table if the file already exist
    if os.path.exists(save_location):
        # Read the CSV file using pandas
        df = pd.read_csv(save_location, delimiter=";")

        # check the size
        dataset_size = df.shape

        # Customize the DataFrame columns and index names
        df.columns = columns

        # Add a new column 'No' with the desired index values
        df.insert(0, "No", range(1, len(df) + 1))

        # Set the 'No' column as the index
        df.set_index("No")

        # Convert the DataFrame to an HTML table
        data_table = df.to_html(index=False)

    return (
        columns,
        error_msg,
        success_msg,
        dataset_size,
        data_table,
    )
