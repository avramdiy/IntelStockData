from flask import Flask, send_file, render_template, render_template_string, jsonify, request
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
from app.model import load_and_preprocess_data, build_lstm_model, train_model, predict_stock_prices, visualize_predictions

app = Flask(__name__, template_folder=r"C:\\Users\\Ev\\Desktop\\IntelStockData\\templates")

data_path = r"C:\\Users\\Ev\\Desktop\\IntelStockData\\data.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    raise FileNotFoundError(f"Data file not found at {data_path}")

(X_train, X_test, y_train, y_test), scaler = load_and_preprocess_data(df)

model = build_lstm_model(X_train)
model = train_model(X_train, y_train, model)

@app.route('/')
def home():
    with open(r"C:\\Users\\Ev\\Desktop\\IntelStockData\\templates\\index.html") as f:
        template = f.read()
    html_table = df.to_html(classes='data', index=False).strip()
    return render_template_string(template, tables=html_table, titles=df.columns.values)

@app.route('/data', methods=['GET'])
def get_data():
    return jsonify(df.head(10).to_dict(orient='records'))

@app.route('/data/filter', methods=['GET'])
def filter_data():
    column = request.args.get('column')
    value = request.args.get('value')

    if column and value:
        if column in df.columns:
            filtered_df = df[df[column] == value]
            return jsonify(filtered_df.to_dict(orient='records'))
        else:
            return jsonify({"error": f"Column '{column}' not found in dataframe."}), 400
    else:
        return jsonify({"error": "Please provide 'column' and 'value' query parameters."})
    

@app.route('/data/summary', methods=['GET'])
def summary_data():
    
    summary = df.describe().to_dict()
    return jsonify(summary)

@app.route('/data/visualize/close', methods=['GET'])
def visualize_data():
    df['Date'] = pd.to_datetime(df['Date'])
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.title('Stock Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.xticks(rotation=45)
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return send_file(img, mimetype='image/png')

@app.route('/data/visualize/volume', methods=['GET'])
def visualize_volume():
    # Ensure that the 'Date' column is properly converted to datetime and set to UTC
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # Generate the Trading Volume Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Volume'], label='Volume', color='orange')
    plt.title('Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)
    plt.legend()

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Return the image as a response
    return send_file(img, mimetype='image/png')

@app.route('/data/visualize/close/download', methods=['GET'])
def download_close_plot():
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    img = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.title('Stock Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png', as_attachment=True, download_name='closing_price_plot.png')

@app.route('/data/visualize/volume/download', methods=['GET'])
def download_volume_plot():
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    img = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Volume'], label='Volume', color='orange')
    plt.title('Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png', as_attachment=True, download_name='volume_plot.png')

@app.route('/data/visualize/predictions', methods=['GET'])
def visualize_predictions_route():
    january_2025_dates = pd.date_range(start='2025-01-01', end='2025-01-31')
    predicted_prices = predict_stock_prices(model, X_test, scaler, january_2025_dates)
    visualize_predictions(predicted_prices, january_2025_dates)
    return "Prediction chart displayed"

if __name__ == '__main__':
    app.run(debug=True)