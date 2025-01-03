# Data Interactivity for Intel Stock Data (1980 - 2024)

- This project is a Flask API designed to interact with a dataset located at https://www.kaggle.com/datasets/mhassansaboor/intel-stock-data-1980-2024

- The initial commit focuses on setting up the foundational API functionality and loading the CSV data into a pandas DataFrame.

## First Commit:

Loading the Data:

- The CSV file is loaded using pandas (pd.read_csv) from the specified path.

- Includes error handling to ensure the file exists before attempting to load it.

### API Endpoints:

- /data: Returns the first 10 rows of the dataset in JSON format.

- /data/filter: Allows filtering of the dataset based on a specified column and value provided as query parameters.

- /data/summary: Provides summary statistics for numerical columns in the dataset.

### Additional Files

- Added requirements.txt

- Added README.md

## Second Commit

- The / route was modified to load the IMDb dataset into a Pandas DataFrame, made scrollable with CSS.

- Initialized the index.html file, in which CSS was added for functionality.

## Third Commit

- For the first two visualizations, we used matplotlib.pyplot and BytesIO to save PNG files.

- Initialized the /data/visualize/close route, which generates a line plot of the closing price over time.

- Initialized the /data/visualize/volume route, which generates a line plot of the trading volume over time.

## Fourth Commit

- Initialized two individual download routes to save the aforementioned charts to local storage.

## Fifth Commit

- Built and initialized a Long Short-Term Memory model for Recurrent Neural Network Machine Learning focused towards the predicted closing stock price of Intel for the month of January 2025, based on data from 1980-2024.

- To execute the model and display the prediction:

- "python -m app.model" to build the RNN model

- "python -m app.data" to boot the Flask API

- "/data/visualize/predictions route in browser to access prediction chart"

- Integrated Tensorflow, Keras, & Matplotlib for final analysis.