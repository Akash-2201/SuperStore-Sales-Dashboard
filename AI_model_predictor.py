import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

print("1. Loading your new data file...")

try:
    # Load your dataset
    df = pd.read_excel("Dataset/Sample - Superstore.xls")

except FileNotFoundError:
    print("Error: File not found. Check the file path!")
    exit()

# --- STEP 1: PREPROCESSING ---
print(" - Cleaning dates and formatting...")

# Convert Order Date
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Month_Year'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()

# Group data
data = df.groupby(['Month_Year', 'Sub-Category', 'Region'])[['Sales', 'Profit']].sum().reset_index()

# Create time features
data['Month'] = data['Month_Year'].dt.month
data['Year'] = data['Month_Year'].dt.year

# Convert text columns to numbers
le_cat = LabelEncoder()
data['Sub-Category_Code'] = le_cat.fit_transform(data['Sub-Category'])

le_reg = LabelEncoder()
data['Region_Code'] = le_reg.fit_transform(data['Region'])

features = ['Month', 'Year', 'Sub-Category_Code', 'Region_Code']
X = data[features]

# --- STEP 2: TRAINING THE AI ---
print("2. Training the AI Models...")

# MODEL 1: Sales Prediction
print(" - Training Sales Model...")
model_sales = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5
)

model_sales.fit(X, data['Sales'])

# MODEL 2: Profit Prediction
print(" - Training Profit/Loss Model...")
model_profit = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5
)

model_profit.fit(X, data['Profit'])

# --- STEP 3: PREDICTING THE FUTURE ---
print("3. Forecasting the next 24 Months...")

last_date = data['Month_Year'].max()

# Generate future dates (24 months)
future_dates = [last_date + pd.DateOffset(months=x+1) for x in range(24)]

future_rows = []
unique_cats = data['Sub-Category'].unique()
unique_regions = data['Region'].unique()

# Create scenarios
for date in future_dates:
    for cat in unique_cats:
        for reg in unique_regions:
            future_rows.append({
                'Month_Year': date,
                'Sub-Category': cat,
                'Region': reg,
                'Month': date.month,
                'Year': date.year
            })

future_df = pd.DataFrame(future_rows)

# Encode future data
future_df['Sub-Category_Code'] = le_cat.transform(future_df['Sub-Category'])
future_df['Region_Code'] = le_reg.transform(future_df['Region'])

# Make predictions
future_X = future_df[features]

future_df['Sales'] = model_sales.predict(future_X)
future_df['Profit'] = model_profit.predict(future_X)
future_df['Type'] = 'Forecast'

# --- STEP 4: SAVING THE RESULT ---
print("4. Saving final file...")

# Historical data
history_df = data[['Month_Year', 'Sub-Category', 'Region', 'Sales', 'Profit']].copy()
history_df['Type'] = 'Historical'

# Combine past + future
final_df = pd.concat([
    history_df,
    future_df[['Month_Year', 'Sub-Category', 'Region', 'Sales', 'Profit', 'Type']]
])

final_df.rename(columns={'Month_Year': 'Order Date'}, inplace=True)

# Export file
output_filename = 'final_ai_prediction1.csv'
final_df.to_csv(output_filename, index=False)

print(f"\nSUCCESS! File '{output_filename}' is ready.")
print(" - This file contains REAL historical Profit and AI-Predicted Future Profit.")
print(" - Load this into Power BI and create your Line Charts!")
