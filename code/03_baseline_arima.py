# 03_baseline_arima.py

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# load the raw dataset created in step 01
data_path = "../data/raw/synthetic_energy_data.csv"
data = pd.read_csv(data_path)

# select only the target variable for ARIMA
energy_demand = data["energy_demand"].values

# define split index
train_size = int(len(energy_demand) * 0.8)

# split without shuffling
train_data = energy_demand[:train_size]
test_data = energy_demand[train_size:]

# fit ARIMA model
# order is kept simple on purpose
model = ARIMA(train_data, order=(2, 1, 2))
model_fit = model.fit()

# forecast for test period
forecast_steps = len(test_data)
forecast = model_fit.forecast(steps=forecast_steps)

# calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(test_data, forecast))
mae = mean_absolute_error(test_data, forecast)
mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

print("ARIMA baseline evaluation")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"MAPE: {mape:.2f}%")

"""The ARIMA model shows relatively high error values because it operates on a single time series and cannot incorporate exogenous variables or long-term seasonal dependencies present in the data."""