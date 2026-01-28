#01_data_generation.py
import numpy as np
import pandas as pd

# For reproducibility
np.random.seed(42)

# Configuration
N_TIMESTEPS = 1800          # total observations (> 1000 as required)
START_DATE = "2018-01-01"

# Time index
dates = pd.date_range(start=START_DATE, periods=N_TIMESTEPS, freq="D")
t = np.arange(N_TIMESTEPS)

# Feature 1: Temperature
# Yearly seasonality + small noise
temp_yearly = 10 * np.sin(2 * np.pi * t / 365)
temp_noise = np.random.normal(0, 1.5, N_TIMESTEPS)
temperature = 25 + temp_yearly + temp_noise

# Feature 2: Humidity
# Weekly seasonality
humidity_weekly = 8 * np.sin(2 * np.pi * t / 7)
humidity_noise = np.random.normal(0, 2, N_TIMESTEPS)
humidity = 60 + humidity_weekly + humidity_noise

# Feature 3: Industrial load
# Trend + volatility clustering
trend = 0.01 * t
volatility = np.zeros(N_TIMESTEPS)
volatility[0] = 1.0
for i in range(1, N_TIMESTEPS):
    volatility[i] = 0.85 * volatility[i - 1] + np.random.normal(0, 0.5)
industrial_load = 50 + trend + volatility + np.random.normal(0, 1, N_TIMESTEPS)

#Target: Energy_demand
# Depends on past seasonal patterns
energy_demand = (
    0.5 * temperature +
    0.3 * humidity +
    0.4 * industrial_load +
    5 * np.sin(2 * np.pi * t / 365) +
    3 * np.sin(2 * np.pi * t / 7) +
    np.random.normal(0, 2, N_TIMESTEPS))

#Create DataFrame
data = pd.DataFrame({
    "date": dates,
    "temperature": temperature,
    "humidity": humidity,
    "industrial_load": industrial_load,
    "energy_demand": energy_demand})

import os
# Save dataset
output_dir = "../data/raw"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "synthetic_energy_data.csv")
data.to_csv(output_path, index=False)

print("Synthetic dataset generated successfully.")
print(f"Saved to: {output_path}")
print(data.head())
