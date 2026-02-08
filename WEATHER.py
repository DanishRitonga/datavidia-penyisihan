# %% [markdown]
# # Weather Data Pipeline
# Combine all daily weather data from cuaca-harian folder (5 stations) and extend to full daily range

# %% Imports and Setup
import pandas as pd
import numpy as np
from pathlib import Path
import re
from glob import glob

pd.set_option('display.max_columns', None)

# %% Configuration
weather_folder = Path('data/cuaca-harian')
output_file = 'data/cleaned/weather_2010-2025.csv'

print("="*60)
print("WEATHER DATA PIPELINE")
print("="*60)

# %% Define Functions

def extract_station_id(filename):
    """Extract station ID from filename (dki1 -> DKI1, etc.)"""
    basename = Path(filename).name.lower()
    match = re.search(r'dki(\d)', basename)
    if match:
        return f'DKI{match.group(1)}'
    return None

# %% Load Weather Files
print("\n" + "="*60)
print("LOADING WEATHER FILES")
print("="*60)

# Get all weather CSV files from cuaca-harian folder
weather_files = sorted(weather_folder.glob('*.csv'))

print(f"\nFound {len(weather_files)} weather files:")
for file in weather_files:
    station = extract_station_id(file)
    print(f"  - {file.name} → {station}")

# %% Combine All Weather Files
print("\n" + "="*60)
print("COMBINING DATASETS")
print("="*60)

all_dfs = []

for file in weather_files:
    station_id = extract_station_id(file)
    
    # Load CSV file
    df = pd.read_csv(file)
    
    # Add station column
    df['stasiun'] = station_id
    
    # Rename 'time' to 'tanggal' for consistency
    df = df.rename(columns={'time': 'tanggal'})
    
    # Convert to datetime
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    
    all_dfs.append(df)
    print(f"  Loaded {len(df)} rows from {station_id}")

# Combine all dataframes
combined_df = pd.concat(all_dfs, ignore_index=True)

print(f"\nTotal rows: {len(combined_df):,}")
print(f"Date range: {combined_df['tanggal'].min()} to {combined_df['tanggal'].max()}")
print(f"Unique stations: {sorted(combined_df['stasiun'].unique())}")

# %% Extend to Full Daily Range
print("\n" + "="*60)
print("EXTENDING TO FULL DAILY RANGE")
print("="*60)

# Create full daily date range
daily_index = pd.date_range(start='2010-01-01', end='2025-08-31', freq='D')
stations = sorted(combined_df['stasiun'].unique())

print(f"\nCreating daily skeleton:")
print(f"  Date range: {daily_index.min()} to {daily_index.max()}")
print(f"  Total days: {len(daily_index):,}")
print(f"  Stations: {stations}")

# Get list of weather columns (exclude tanggal and stasiun)
weather_cols = [col for col in combined_df.columns if col not in ['tanggal', 'stasiun']]

# Process each station separately
extended_dfs = []

for station in stations:
    print(f"\nProcessing {station}...")
    
    # Get data for this station
    station_data = combined_df[combined_df['stasiun'] == station].copy()
    station_data = station_data.sort_values('tanggal').set_index('tanggal')
    
    # Get actual observation dates (before forward fill)
    obs_dates = station_data.index.copy()
    
    # Reindex to daily range and forward fill
    station_daily = station_data.reindex(daily_index)
    station_daily[weather_cols] = station_daily[weather_cols].ffill()
    
    # Backfill any remaining NaNs at the beginning
    station_daily[weather_cols] = station_daily[weather_cols].bfill()
    
    # Add stasiun column
    station_daily['stasiun'] = station
    
    # Calculate days since last weather update
    station_daily['days_since_weather_update'] = station_daily.index.to_series().apply(
        lambda d: (d - obs_dates[obs_dates <= d].max()).days if any(obs_dates <= d) else -1
    )
    
    # Reset index to make tanggal a column
    station_daily = station_daily.reset_index().rename(columns={'index': 'tanggal'})
    
    extended_dfs.append(station_daily)
    
    print(f"  Original rows: {len(station_data):,}")
    print(f"  Extended rows: {len(station_daily):,}")
    print(f"  Unique observation dates: {len(obs_dates)}")

# Combine all stations
weather_daily = pd.concat(extended_dfs, ignore_index=True)

print(f"\n✓ Extended dataset created")
print(f"  Total rows: {len(weather_daily):,}")

# %% Create ID Column
print("\n" + "="*60)
print("CREATING ID COLUMN")
print("="*60)

# Create ID column in format YYYY-MM-DD_DKIx
weather_daily['ID'] = weather_daily['tanggal'].dt.strftime('%Y-%m-%d') + '_' + weather_daily['stasiun']

print(f"✓ Created ID column")
print(f"  Format: YYYY-MM-DD_DKIx")
print(f"  Unique IDs: {weather_daily['ID'].nunique():,}")

# Check for duplicates
duplicates = weather_daily[weather_daily.duplicated('ID', keep=False)]
if len(duplicates) > 0:
    print(f"\nWARNING: Found {len(duplicates)} duplicate IDs")
else:
    print(f"  No duplicate IDs found ✓")

# %% Reorder Columns
print("\n" + "="*60)
print("FINALIZING DATASET")
print("="*60)

# Convert tanggal back to string for consistency
weather_daily['tanggal'] = weather_daily['tanggal'].dt.strftime('%Y-%m-%d')

# Reorder columns to put ID, tanggal, and stasiun near the front
column_order = ['ID', 'tanggal', 'stasiun', 'days_since_weather_update'] + weather_cols
weather_daily = weather_daily[column_order]

# Sort by date and station
weather_daily = weather_daily.sort_values(['tanggal', 'stasiun']).reset_index(drop=True)

print(f"Final dataset shape: {weather_daily.shape}")
print(f"  Total rows: {len(weather_daily):,}")
print(f"  Total columns: {len(weather_daily.columns)}")

# %% Data Quality Report
print("\n" + "="*60)
print("DATA QUALITY REPORT")
print("="*60)

print(f"\nDate range: {weather_daily['tanggal'].min()} to {weather_daily['tanggal'].max()}")
print(f"Stations: {sorted(weather_daily['stasiun'].unique())}")

print("\nRows by station:")
station_counts = weather_daily['stasiun'].value_counts().sort_index()
for station, count in station_counts.items():
    print(f"  {station}: {count:,} rows")

print("\nNULL values by column:")
null_summary = weather_daily.isnull().sum()
null_summary = null_summary[null_summary > 0]
if len(null_summary) > 0:
    for col, count in null_summary.items():
        pct = (count / len(weather_daily)) * 100
        print(f"  {col}: {count:,} ({pct:.2f}%)")
else:
    print("  No NULL values found ✓")

print("\nData staleness summary:")
print(f"  Mean days since update: {weather_daily['days_since_weather_update'].mean():.1f}")
print(f"  Max days since update: {weather_daily['days_since_weather_update'].max()}")
print(f"  Rows with fresh data (0 days): {(weather_daily['days_since_weather_update'] == 0).sum():,}")

# %% Display Sample Data
print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)

print("\nFirst 10 rows:")
sample_cols = ['ID', 'tanggal', 'stasiun', 'days_since_weather_update', 
               'temperature_2m_max (°C)', 'precipitation_sum (mm)', 'wind_speed_10m_max (km/h)']
available_sample_cols = [col for col in sample_cols if col in weather_daily.columns]
print(weather_daily[available_sample_cols].head(10).to_string())

print("\n\nLast 10 rows:")
print(weather_daily[available_sample_cols].tail(10).to_string())

print("\n\nData from each station (recent sample):")
for station in sorted(weather_daily['stasiun'].unique()):
    station_data = weather_daily[weather_daily['stasiun'] == station].iloc[-1]
    print(f"\n{station} (latest date: {station_data['tanggal']}):")
    print(f"  Days since update: {station_data['days_since_weather_update']}")
    if 'temperature_2m_max (°C)' in weather_daily.columns:
        print(f"  Max Temp: {station_data['temperature_2m_max (°C)']}°C")
    if 'precipitation_sum (mm)' in weather_daily.columns:
        print(f"  Precipitation: {station_data['precipitation_sum (mm)']}mm")

# %% Save to CSV
print("\n" + "="*60)
print("SAVING DATA")
print("="*60)

# Create output directory if it doesn't exist
output_path = Path(output_file)
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save to CSV
weather_daily.to_csv(output_file, index=False)
print(f"\n✓ Data saved to: {output_file}")
print(f"  Total rows: {len(weather_daily):,}")
print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)
