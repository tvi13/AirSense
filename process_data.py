import requests
import pandas as pd
import numpy as np
import datetime
import os

# Resolve all paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sensor node identifiers — must match column naming in the CSV
nodes = ['KE', 'K', 'VPW', 'D', 'C', 'BKC', 'CE', 'SN', 'CSIA', 'S']

# 1. Read sensor data first to dynamically extract date range
print("Reading sensor data...")
csv_path = os.path.join(BASE_DIR, "Mumbai_AQI_Cleaned.csv")
df = pd.read_csv(csv_path)

# Extract dynamic min/max date range from the CSV
df_dates = pd.to_datetime(df['From Date'], format='%d-%m-%Y %H:%M')
start_date_str = df_dates.min().strftime('%Y-%m-%d')
end_date_str = df_dates.max().strftime('%Y-%m-%d')

# 2. Fetch data from Open-Meteo API
print(f"Fetching weather data from Open-Meteo ({start_date_str} to {end_date_str})...")
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 19.0760,
    "longitude": 72.8777,
    "start_date": start_date_str,
    "end_date": end_date_str,
    "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
    "timezone": "Asia/Kolkata"
}
response = requests.get(url, params=params)
response.raise_for_status()
data = response.json()

hourly = data['hourly']
weather_df = pd.DataFrame({
    'time': hourly['time'],
    'Temperature': hourly['temperature_2m'],
    'Wind Speed': hourly['wind_speed_10m'],
    'Wind Direction': hourly['wind_direction_10m']
})

# Open-Meteo formats time as "YYYY-MM-DDTHH:MM"
# We need to map it to "DD-MM-YYYY HH:MM" format in the CSV
weather_df['time_formatted'] = pd.to_datetime(weather_df['time']).dt.strftime('%d-%m-%Y %H:%M')

# 3. Clean column names
print("Cleaning column names...")
df.columns = [c.replace('(', '').strip() for c in df.columns]

# 4. Merge data
print("Merging weather data...")
df = df.merge(weather_df, left_on='From Date', right_on='time_formatted', how='left')

# 5. Extract 3D Tensor
print("Structuring 3D NumPy array...")
# Features: [PM2.5, PM10, NO2, CO, Temp, WindSpeed, WindDir]
features = ['PM2.5', 'PM10', 'NO2', 'CO']

num_hours = len(df)
num_nodes = len(nodes)
num_features = len(features) + 3 # 4 sensor + 3 weather

tensor = np.zeros((num_hours, num_nodes, num_features), dtype=np.float32)

for i, node in enumerate(nodes):
    node_cols = [f"{node}_{feat}" for feat in features]
    # Copy sensor data
    tensor[:, i, :4] = df[node_cols].values
    # Copy weather data
    tensor[:, i, 4] = df['Temperature'].values
    tensor[:, i, 5] = df['Wind Speed'].values
    tensor[:, i, 6] = df['Wind Direction'].values

print(f"Final tensor shape: {tensor.shape}")

# Save the PyTorch-ready tensor
out_path = os.path.join(BASE_DIR, "mumbai_tensor.npy")
np.save(out_path, tensor)
print(f"Tensor successfully saved to {out_path}")

# 6. Save a 2D flattened version for human-readability
print("Saving 2D flattened CSV version...")
flat_tensor = tensor.reshape(num_hours, -1)

# Create 2D column names
flat_cols = []
for node in nodes:
    for feat in features + ['Temperature', 'Wind Speed', 'Wind Direction']:
        flat_cols.append(f"{node}_{feat}")

flat_df = pd.DataFrame(flat_tensor, columns=flat_cols)
flat_df.insert(0, 'Hour_Index', range(num_hours))
csv_out_path = os.path.join(BASE_DIR, "mumbai_tensor_flattened.csv")
flat_df.to_csv(csv_out_path, index=False)
print(f"Flattened CSV successfully saved to {csv_out_path}")
