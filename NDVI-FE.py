#%% imports
import pandas as pd
import numpy as np
from pathlib import Path

#%% Load daily NDVI data
print("="*60)
print("NDVI FEATURE ENGINEERING")
print("="*60)

input_file = Path('data/cleaned/ndvi_2010_2025-v2.csv')
output_file = Path('data/cleaned/ndvi_with_features.csv')

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

# 1. Lag features (historical NDVI values)
print("\n1. Creating lag features...")
df['ndvi_lag_7d'] = grouped['ndvi'].shift(7)
df['ndvi_lag_14d'] = grouped['ndvi'].shift(14)
df['ndvi_lag_30d'] = grouped['ndvi'].shift(30)
df['ndvi_lag_60d'] = grouped['ndvi'].shift(60)
print("   ✓ ndvi_lag_7d, ndvi_lag_14d, ndvi_lag_30d, ndvi_lag_60d created")

# 2. Rolling mean features (smoothed trends)
print("\n2. Creating rolling mean features...")
df['ndvi_rollmean_14d'] = grouped['ndvi'].transform(
    lambda x: x.rolling(window=14, min_periods=1).mean()
)
df['ndvi_rollmean_30d'] = grouped['ndvi'].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
)
print("   ✓ ndvi_rollmean_14d, ndvi_rollmean_30d created")

# 3. Delta feature (change from 30 days ago)
print("\n3. Creating delta feature...")
df['ndvi_delta_30d'] = df['ndvi'] - df['ndvi_lag_30d']
print("   ✓ ndvi_delta_30d created (ndvi - ndvi_lag_30d)")

# 4. Additional useful features (bonus)
print("\n4. Creating additional features...")

# Rolling standard deviation (volatility)
df['ndvi_rollstd_30d'] = grouped['ndvi'].transform(
    lambda x: x.rolling(window=30, min_periods=7).std()
)

# Short-term deltas
df['ndvi_delta_7d'] = df['ndvi'] - df['ndvi_lag_7d']
df['ndvi_delta_14d'] = df['ndvi'] - df['ndvi_lag_14d']

# Percentage changes
df['ndvi_pct_change_30d'] = grouped['ndvi'].pct_change(periods=30)

# Deviation from rolling mean
df['ndvi_deviation_from_30d_mean'] = df['ndvi'] - df['ndvi_rollmean_30d']

print("   ✓ Additional features created")

#%% Feature summary
print("\n" + "="*60)
print("FEATURE SUMMARY")
print("="*60)

feature_cols = [
    'ndvi_lag_7d',
    'ndvi_lag_14d',
    'ndvi_lag_30d',
    'ndvi_lag_60d',
    'ndvi_rollmean_14d',
    'ndvi_rollmean_30d',
    'ndvi_delta_30d',
    'ndvi_rollstd_30d',
    'ndvi_delta_7d',
    'ndvi_delta_14d',
    'ndvi_pct_change_30d',
    'ndvi_deviation_from_30d_mean'
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
base_cols = ['tanggal', 'stasiun', 'ndvi', 'days_since_ndvi_update']
final_cols = base_cols + feature_cols

df_final = df[final_cols].copy()

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Columns: {len(final_cols)}")

#%% Display sample data
print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)

print("\nFirst 15 rows (showing lag features develop over time):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 4)
print(df_final.head(15).to_string())

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
    print(f"  NDVI range: {station_data['ndvi'].min():.4f} to {station_data['ndvi'].max():.4f}")
    print(f"  Mean NDVI: {station_data['ndvi'].mean():.4f}")
    print(f"  Mean 30-day delta: {station_data['ndvi_delta_30d'].mean():.6f}")
    print(f"  Max 30-day increase: {station_data['ndvi_delta_30d'].max():.4f}")
    print(f"  Max 30-day decrease: {station_data['ndvi_delta_30d'].min():.4f}")

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
print("FEATURE CORRELATIONS WITH NDVI")
print("="*60)

correlations = df_final[feature_cols + ['ndvi']].corr()['ndvi'].drop('ndvi').sort_values(ascending=False)
print("\nTop features correlated with NDVI:")
for feat, corr in correlations.items():
    print(f"  {feat:35s}: {corr:7.4f}")
