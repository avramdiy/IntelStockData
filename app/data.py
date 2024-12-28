from flask import Flask, render_template, render_template_string, jsonify, request
import pandas as pd
import os

app = Flask(__name__, template_folder=r"C:\\Users\\Ev\\Desktop\\IntelStockData\\templates")

data_path = r"C:\\Users\\Ev\\Desktop\\IntelStockData\\data.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    raise FileNotFoundError(f"Data file not found at {data_path}")

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

if __name__ == '__main__':
    app.run(debug=True)