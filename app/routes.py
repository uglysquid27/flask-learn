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
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("SELECT do_date, device_name, mst_history.value FROM mst_history WHERE area_name = 'OCI1' AND device_name = 'CAP - FEEDER C/V 1' AND test_name = '2H'")

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        df = pd.DataFrame(rows, columns=['Date', 'Device_Name', 'Value'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        df.index = df.index.strftime('%Y-%m-%d')

        forecast_steps = len(df) 
        model = ARIMA(df['Value'], order=(1, 1, 0))
        results = model.fit()
        forecast = results.forecast(steps=forecast_steps)

        forecast_values = forecast.tolist()

        # Calculate MAPE if forecasted values and actual values have the same length
        if len(forecast_values) == len(df):
            actual_values = df['Value'].values
            mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
        else:
            mape = None  # Set MAPE to None if lengths do not match

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

@app.route('/montecarlo')
def monte_carlo_simulation():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT do_date, device_name, mst_history.value FROM mst_history WHERE area_name = 'OCI1' AND device_name = 'CAP - FEEDER C/V 1' AND test_name = '2H'")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        df = pd.DataFrame(rows, columns=['Date', 'Device_Name', 'Value'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        df.index = df.index.strftime('%Y-%m-%d')

        monte_carlo_simulations = 1000 
        simulation_results = []
        for _ in range(monte_carlo_simulations):
            noise = np.random.normal(0, 1, len(df))
            simulated_values = df['Value'] + noise
            simulation_results.append(simulated_values.tolist())

       
        actual_values = df['Value'].values
        mape_monte_carlo = []
        for sim_values in simulation_results:
            mape_value = np.mean(np.abs((actual_values - sim_values) / actual_values)) * 100
            mape_monte_carlo.append(mape_value)

        response_data = {
            'simulation_results': simulation_results,
            'mape_monte_carlo': mape_monte_carlo,
            'data': df.to_dict(orient='records')
        }

        return jsonify(response_data)

    except Exception as e:
        error_message = f"Failed to fetch data from the database or perform Monte Carlo simulation: {str(e)}"
        print(error_message)
        return jsonify({'error': error_message})

if __name__ == '__main__':
    app.run(debug=True)