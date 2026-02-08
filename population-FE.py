#%% imports
import pandas as pd
import numpy as np
from pathlib import Path

#%% Load daily population data
print("="*60)
print("POPULATION FEATURE ENGINEERING")
print("="*60)

input_file = Path('populasi_2010_2025_daily_long.csv')
output_file = Path('data/cleaned/population_with_features.csv')

print(f"\nLoading data from: {input_file}")
df = pd.read_csv(input_file)
df['tanggal'] = pd.to_datetime(df['tanggal'])

print(f"Loaded {len(df):,} rows")
print(f"Date range: {df['tanggal'].min()} to {df['tanggal'].max()}")
print(f"Stations: {sorted(df['stasiun'].unique())}")

#%% Sort and prepare for feature engineering
df = df.sort_values(['stasiun', 'tanggal']).reset_index(drop=True)

#%% Feature Engineering
print("\n" + "="*60)
print("CREATING FEATURES")
print("="*60)

# Group by station for time-series features
grouped = df.groupby('stasiun', group_keys=False)

# 1. Log population (helps with scale in regression models)
print("\n1. Creating log population feature...")
df['pop_log'] = np.log(df['populasi'])
print("   ✓ pop_log created")

# 2. Population change from 1 year ago (~365 days)
print("\n2. Creating 1-year population change...")
df['pop_change_1y'] = grouped.apply(
    lambda x: x['populasi'].diff(periods=365)
).reset_index(drop=True)
print("   ✓ pop_change_1y created (current - 365 days ago)")

# 3. Population growth rate (percentage change from 1 year ago)
print("\n3. Creating 1-year growth rate...")
df['pop_1y_ago'] = grouped.apply(
    lambda x: x['populasi'].shift(365)
).reset_index(drop=True)

df['pop_growth_rate_1y'] = (df['populasi'] / df['pop_1y_ago']) - 1
df = df.drop('pop_1y_ago', axis=1)  # Drop intermediate column
print("   ✓ pop_growth_rate_1y created (percentage change)")

# 4. Short-term changes (30-day, 90-day)
print("\n4. Creating short-term changes...")
df['pop_change_30d'] = grouped.apply(
    lambda x: x['populasi'].diff(periods=30)
).reset_index(drop=True)

df['pop_change_90d'] = grouped.apply(
    lambda x: x['populasi'].diff(periods=90)
).reset_index(drop=True)
print("   ✓ pop_change_30d and pop_change_90d created")

# 5. Rolling averages (smooths out jumps from census updates)
print("\n5. Creating rolling statistics...")
df['pop_rolling_mean_365d'] = grouped.apply(
    lambda x: x['populasi'].rolling(window=365, min_periods=1).mean()
).reset_index(drop=True)

df['pop_rolling_std_365d'] = grouped.apply(
    lambda x: x['populasi'].rolling(window=365, min_periods=30).std()
).reset_index(drop=True)
print("   ✓ pop_rolling_mean_365d and pop_rolling_std_365d created")

# 6. Momentum features (rate of change of growth rate)
print("\n6. Creating momentum features...")
df['pop_growth_rate_30d'] = grouped.apply(
    lambda x: x['populasi'].pct_change(periods=30)
).reset_index(drop=True)

df['pop_growth_rate_90d'] = grouped.apply(
    lambda x: x['populasi'].pct_change(periods=90)
).reset_index(drop=True)
print("   ✓ pop_growth_rate_30d and pop_growth_rate_90d created")

# 7. Deviation from rolling mean (captures unusual population changes)
print("\n7. Creating deviation features...")
df['pop_deviation_from_365d_mean'] = df['populasi'] - df['pop_rolling_mean_365d']
df['pop_deviation_pct'] = (df['pop_deviation_from_365d_mean'] / df['pop_rolling_mean_365d']) * 100
print("   ✓ Deviation features created")

#%% Feature summary
print("\n" + "="*60)
print("FEATURE SUMMARY")
print("="*60)

feature_cols = [
    'pop_log',
    'pop_change_1y',
    'pop_growth_rate_1y',
    'pop_change_30d',
    'pop_change_90d',
    'pop_rolling_mean_365d',
    'pop_rolling_std_365d',
    'pop_growth_rate_30d',
    'pop_growth_rate_90d',
    'pop_deviation_from_365d_mean',
    'pop_deviation_pct'
]

print("\nFeatures created:")
for i, col in enumerate(feature_cols, 1):
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df)) * 100
    print(f"  {i:2d}. {col:35s} - {null_count:6,} nulls ({null_pct:5.2f}%)")

#%% Column ordering
print("\n" + "="*60)
print("FINALIZING DATASET")
print("="*60)

# Reorder columns for clarity
base_cols = ['tanggal', 'stasiun', 'populasi', 'days_since_pop_update']
final_cols = base_cols + feature_cols

df_final = df[final_cols].copy()

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Columns: {len(final_cols)}")

#%% Display sample data
print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)

print("\nFirst 10 rows (early data with new features):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df_final.head(10).to_string())

print("\n\nRecent data sample (2024-2025):")
recent_data = df_final[df_final['tanggal'] >= '2024-01-01'].head(10)
print(recent_data.to_string())

#%% Statistics by station
print("\n" + "="*60)
print("STATISTICS BY STATION")
print("="*60)

for station in sorted(df_final['stasiun'].unique()):
    station_data = df_final[df_final['stasiun'] == station]
    print(f"\n{station}:")
    print(f"  Population range: {station_data['populasi'].min():,.0f} to {station_data['populasi'].max():,.0f}")
    print(f"  Mean 1-year growth rate: {station_data['pop_growth_rate_1y'].mean()*100:.2f}%")
    print(f"  Max 1-year change: {station_data['pop_change_1y'].max():,.0f}")
    print(f"  Min 1-year change: {station_data['pop_change_1y'].min():,.0f}")

#%% Save to CSV
print("\n" + "="*60)
print("SAVING DATA")
print("="*60)

# Create output directory if it doesn't exist
output_file.parent.mkdir(parents=True, exist_ok=True)

# Save to CSV
df_final.to_csv(output_file, index=False)
print(f"\n✓ Data saved to: {output_file}")
print(f"  Total rows: {len(df_final):,}")
print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")

print("\n" + "="*60)
print("FEATURE ENGINEERING COMPLETED")
print("="*60)

#%% Quick correlation analysis (optional)
print("\n" + "="*60)
print("FEATURE CORRELATIONS WITH POPULATION")
print("="*60)

correlations = df_final[feature_cols + ['populasi']].corr()['populasi'].drop('populasi').sort_values(ascending=False)
print("\nTop features correlated with population:")
for feat, corr in correlations.items():
    print(f"  {feat:35s}: {corr:7.4f}")
