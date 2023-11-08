import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start_date, end_date)
def process_batch(batch, process_id):
    required_columns = ['temperature', 'humidity', 'wind_speed', 'pressure']
    if not all(col in batch.columns for col in required_columns):
        return None

    try:
        X = batch[['temperature', 'humidity', 'wind_speed']]
        y = batch['pressure']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)

        # Extract the rows of data
        batch_data = batch[['temperature', 'humidity', 'wind_speed', 'pressure']]

        # Predict climate for next year
        next_year_predictions = model.predict(X)

        # Store results in a dictionary
        results = {
            'process_id': process_id,
            'mse': mse,
            'next_year_predictions': next_year_predictions.tolist(),
            'batch_data': batch_data.values.tolist()  # Convert the DataFrame to a list of lists
        }

        return results
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None

results_list = []

def process_batch_with_core(core_id, batch):
    global results_list

    process_id = core_id % 4 + 1
    results = process_batch(batch, process_id)

    if results is not None:
        results_list.append(results)
        print(f"Batch {core_id + 1} processed in core {process_id} process id: {results['process_id']}")
    else:
        print("")

if __name__ == '__main__':
    results_list = []
    unique_process_ids = []  # List to store unique process IDs

    data = pd.read_csv('big_data.csv')

    batch_size = 100
    batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    num_cores = min(multiprocessing.cpu_count(), 4)

    pool = multiprocessing.Pool(num_cores)

    pool.starmap(process_batch_with_core, enumerate(batches))

    pool.close()
    pool.join()

    # Generate unique process IDs
    for result in results_list:
        if result is not None:
            unique_process_ids.append(result['process_id'])

    results_df = pd.DataFrame([row for row in results_list if row is not None])

    # Save the unique process IDs to results.csv
    unique_process_id_df = pd.DataFrame({'process_id': unique_process_ids})
    unique_process_id_df.to_csv('results.csv', index=False)

    # Save the rest of the results to results.csv
    results_df.to_csv('results.csv', mode='a', index=False, header=False)
data = {
    'date': date_range,
    'temperature': np.random.uniform(0, 40, len(date_range)),
    'humidity': np.random.uniform(20, 90, len(date_range)),  
    'wind_speed': np.random.uniform(0, 20, len(date_range)),
    'pressure': np.random.uniform(900, 1100, len(date_range)),
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('results.csv', index=False)
print("CSV file 'results.csv' has been created.")