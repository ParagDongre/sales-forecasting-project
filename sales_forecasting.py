import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
data = pd.read_csv('sales_data.csv')

# Prepare data
data['Month_Num'] = np.arange(len(data))  # numeric months
X = data[['Month_Num']]
y = data['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Forecast for next 6 months
future_months = np.arange(len(data), len(data) + 6).reshape(-1, 1)
future_sales = model.predict(future_months)

# Save forecast
forecast_df = pd.DataFrame({
    'Month': [f'Month {i}' for i in range(len(data) + 1, len(data) + 7)],
    'Predicted_Sales': future_sales
})
forecast_df.to_csv('sales_forecast.csv', index=False)

# Plot sales trend
plt.figure(figsize=(8, 5))
plt.plot(data['Month'], y, marker='o', label='Actual Sales')
plt.plot(forecast_df['Month'], forecast_df['Predicted_Sales'], marker='x', label='Forecasted Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Forecast')
plt.legend()
plt.savefig('sales_trend.png')
plt.show()
