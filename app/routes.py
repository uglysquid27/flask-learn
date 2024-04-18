from app import app
import pandas as pd
import mysql.connector
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from flask import jsonify
import numpy as np

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'plan_pdm'
}

@app.route('/')
def get_data_from_table():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM mst_history LIMIT 10")

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return '\n'.join(str(row) for row in rows)

    except Exception as e:
        return f"Failed to fetch data from the database: {str(e)}"

@app.route('/arimatest')
def fetch_data_from_database_and_predict():
    try:
        # Connect to the MySQL database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Execute SQL query to fetch data
        cursor.execute("SELECT do_date, device_name, mst_history.value FROM mst_history WHERE area_name = 'OCI1' AND device_name = 'CAP - FEEDER C/V 1' AND test_name = '2H'")

        # Fetch all rows from the query result
        rows = cursor.fetchall()

        # Close cursor and connection
        cursor.close()
        conn.close()

        # Create a pandas DataFrame from the fetched rows
        df = pd.DataFrame(rows, columns=['Date', 'Device_Name', 'Value'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Prepare the DataFrame for ARIMA forecasting
        df.index = df.index.strftime('%Y-%m-%d')

        # Perform ARIMA forecasting
        forecast_steps = len(df)  # Adjust forecast steps to match data length
        model = ARIMA(df['Value'], order=(1, 1, 0))
        results = model.fit()
        forecast = results.forecast(steps=forecast_steps)

        # Convert forecast values to a list
        forecast_values = forecast.tolist()

        # Calculate MAPE if forecasted values and actual values have the same length
        if len(forecast_values) == len(df):
            actual_values = df['Value'].values
            mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
        else:
            mape = None  # Set MAPE to None if lengths do not match

        # Prepare response data
        response_data = {
            'forecast_values': forecast_values,
            'data': df.to_dict(orient='records'),
            'mape': mape
        }

        # Return JSON response
        return jsonify(response_data)

    except Exception as e:
        error_message = f"Failed to fetch data from the database or perform ARIMA prediction: {str(e)}"
        print(error_message)
        return jsonify({'error': error_message})

if __name__ == '__main__':
    app.run(debug=True)