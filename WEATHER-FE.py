#%% imports
import pandas as pd
import numpy as np
from pathlib import Path

#%% Load daily weather data
print("="*60)
print("WEATHER FEATURE ENGINEERING")
print("="*60)

input_file = Path('data/cleaned/weather_2010-2025.csv')
output_file = Path('data/cleaned/weather_with_features.csv')

print(f"\nLoading data from: {input_file}")
df = pd.read_csv(input_file)
df['tanggal'] = pd.to_datetime(df['tanggal'])

print(f"Loaded {len(df):,} rows")
print(f"Date range: {df['tanggal'].min()} to {df['tanggal'].max()}")
print(f"Stations: {sorted(df['stasiun'].unique())}")
print(f"\nAvailable columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

#%% Sort and prepare for feature engineering
df = df.sort_values(['stasiun', 'tanggal']).reset_index(drop=True)

# Identify weather metric columns (exclude meta columns)
meta_cols = ['ID', 'tanggal', 'stasiun', 'days_since_weather_update']
weather_metric_cols = [col for col in df.columns if col not in meta_cols]

print(f"\nWeather metric columns ({len(weather_metric_cols)}):")
for col in weather_metric_cols:
    print(f"  - {col}")

#%% Helper function to clean column names for features
def clean_col_name(col):
    """Convert column name to feature-friendly format"""
    # Remove units and special characters
    col = col.replace(' (°C)', '').replace(' (mm)', '').replace(' (km/h)', '')
    col = col.replace(' (%)', '').replace(' (°)', '').replace(' (hPa)', '')
    col = col.replace(' (MJ/m²)', '').replace(' (mm)', '').replace(' (kWh/m²)', '')
    col = col.replace('_', '_').replace(' ', '_').lower()
    return col

#%% Feature Engineering
print("\n" + "="*60)
print("CREATING FEATURES")
print("="*60)

# Group by station for time-series features
grouped = df.groupby('stasiun', group_keys=False)

# Create a mapping of clean column names
col_mapping = {col: clean_col_name(col) for col in weather_metric_cols}

#%% 1. LAG FEATURES
print("\n1. Creating lag features (7, 14, 30 days)...")

lag_periods = [7, 14, 30]
for col in weather_metric_cols[:3]:  # Create lags for key metrics only to save space
    clean_name = col_mapping[col]
    for lag in lag_periods:
        df[f'{clean_name}_lag_{lag}d'] = grouped[col].shift(lag)

print(f"   ✓ Created lag features for: {', '.join([col_mapping[c] for c in weather_metric_cols[:3]])}")

#%% 2. ROLLING STATISTICS
print("\n2. Creating rolling statistics (7, 14, 30 days)...")

# Temperature rolling stats
if 'temperature_2m_mean (°C)' in df.columns:
    temp_col = 'temperature_2m_mean (°C)'
    df['temp_rollmean_7d'] = grouped[temp_col].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['temp_rollmean_14d'] = grouped[temp_col].transform(lambda x: x.rolling(14, min_periods=1).mean())
    df['temp_rollmean_30d'] = grouped[temp_col].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df['temp_rollstd_30d'] = grouped[temp_col].transform(lambda x: x.rolling(30, min_periods=7).std())
    print("   ✓ Temperature rolling stats created")

# Precipitation rolling stats
if 'precipitation_sum (mm)' in df.columns:
    precip_col = 'precipitation_sum (mm)'
    df['precip_rollsum_7d'] = grouped[precip_col].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df['precip_rollsum_14d'] = grouped[precip_col].transform(lambda x: x.rolling(14, min_periods=1).sum())
    df['precip_rollsum_30d'] = grouped[precip_col].transform(lambda x: x.rolling(30, min_periods=1).sum())
    df['precip_rollmean_30d'] = grouped[precip_col].transform(lambda x: x.rolling(30, min_periods=1).mean())
    print("   ✓ Precipitation rolling stats created")

# Wind speed rolling stats
if 'wind_speed_10m_max (km/h)' in df.columns:
    wind_col = 'wind_speed_10m_max (km/h)'
    df['wind_rollmean_7d'] = grouped[wind_col].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['wind_rollmean_30d'] = grouped[wind_col].transform(lambda x: x.rolling(30, min_periods=1).mean())
    print("   ✓ Wind speed rolling stats created")

#%% 3. DERIVED METRICS
print("\n3. Creating derived weather metrics...")

# Rainy day flag
if 'precipitation_sum (mm)' in df.columns:
    df['is_rainy'] = (df['precipitation_sum (mm)'] > 0).astype('int8')
    print("   ✓ is_rainy flag created")

    # Count rainy days in past 7, 14, 30 days
    df['rainy_days_7d'] = grouped['is_rainy'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df['rainy_days_14d'] = grouped['is_rainy'].transform(lambda x: x.rolling(14, min_periods=1).sum())
    df['rainy_days_30d'] = grouped['is_rainy'].transform(lambda x: x.rolling(30, min_periods=1).sum())
    print("   ✓ Rainy day counts created")

# Wind direction bins
if 'wind_direction_10m_dominant (°)' in df.columns:
    wind_dir_col = 'wind_direction_10m_dominant (°)'
    # Create 8 cardinal direction bins: N, NE, E, SE, S, SW, W, NW
    # Each sector is 45° wide, centered on the cardinal direction
    bins = [-22.5, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
    labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    
    # Handle 360° = 0° wraparound (values > 337.5° are North, adjust to negative)
    df['wind_dir_adjusted'] = df[wind_dir_col].apply(lambda x: x if x <= 337.5 else x - 360)
    df['wind_direction_cardinal'] = pd.cut(df['wind_dir_adjusted'], bins=bins, labels=labels, include_lowest=True)
    df = df.drop('wind_dir_adjusted', axis=1)
    print("   ✓ Wind direction cardinal bins created (N/NE/E/SE/S/SW/W/NW)")

# Heat index approximation
if all(col in df.columns for col in ['temperature_2m_mean (°C)', 'relative_humidity_2m_mean (%)', 'wind_speed_10m_mean (km/h)']):
    temp_mean = df['temperature_2m_mean (°C)']
    humidity_mean = df['relative_humidity_2m_mean (%)']
    wind_mean = df['wind_speed_10m_mean (km/h)']
    
    df['heat_index'] = temp_mean + 0.5 * humidity_mean - 0.1 * wind_mean
    print("   ✓ Heat index approximation created")
elif 'temperature_2m_mean (°C)' in df.columns and 'relative_humidity_2m_mean (%)' in df.columns:
    # Simplified version if wind speed not available
    df['heat_index'] = df['temperature_2m_mean (°C)'] + 0.5 * df['relative_humidity_2m_mean (%)']
    print("   ✓ Heat index approximation created (simplified)")

# Temperature inversion proxy (large diurnal range)
if all(col in df.columns for col in ['temperature_2m_max (°C)', 'temperature_2m_min (°C)']):
    df['temp_range'] = df['temperature_2m_max (°C)'] - df['temperature_2m_min (°C)']
    df['temp_inversion_proxy'] = (df['temp_range'] > 8).astype('int8')  # 8°C threshold
    print("   ✓ Temperature range and inversion proxy created")

# Feels-like temperature (if available)
if 'temperature_2m_mean (°C)' in df.columns and 'wind_speed_10m_mean (km/h)' in df.columns:
    # Simple wind chill approximation
    temp = df['temperature_2m_mean (°C)']
    wind = df['wind_speed_10m_mean (km/h)']
    df['feels_like_temp'] = temp - 0.5 * wind
    print("   ✓ Feels-like temperature created")

# Dry/wet spell indicators
if 'precipitation_sum (mm)' in df.columns:
    # Consecutive dry days
    df['dry_spell'] = grouped['is_rainy'].transform(
        lambda x: (~x.astype(bool)).groupby(x.astype(bool).cumsum()).cumsum()
    )
    # Consecutive wet days
    df['wet_spell'] = grouped['is_rainy'].transform(
        lambda x: x.groupby((~x.astype(bool)).cumsum()).cumsum()
    )
    print("   ✓ Dry/wet spell indicators created")

#%% 4. DELTAS AND CHANGES
print("\n4. Creating delta and change features...")

# Temperature changes
if 'temperature_2m_mean (°C)' in df.columns:
    temp_col = 'temperature_2m_mean (°C)'
    df['temp_delta_1d'] = grouped[temp_col].diff(1)
    df['temp_delta_7d'] = grouped[temp_col].diff(7)
    df['temp_delta_30d'] = grouped[temp_col].diff(30)
    df['temp_change_rate_7d'] = grouped[temp_col].pct_change(7)
    print("   ✓ Temperature deltas created")

# Precipitation changes
if 'precipitation_sum (mm)' in df.columns:
    precip_col = 'precipitation_sum (mm)'
    df['precip_delta_1d'] = grouped[precip_col].diff(1)
    df['precip_delta_7d'] = grouped[precip_col].diff(7)
    print("   ✓ Precipitation deltas created")

# Wind speed changes
if 'wind_speed_10m_max (km/h)' in df.columns:
    wind_col = 'wind_speed_10m_max (km/h)'
    df['wind_delta_7d'] = grouped[wind_col].diff(7)
    df['wind_change_rate_7d'] = grouped[wind_col].pct_change(7)
    print("   ✓ Wind speed deltas created")

# Pressure changes (if available)
if 'surface_pressure_mean (hPa)' in df.columns:
    pressure_col = 'surface_pressure_mean (hPa)'
    df['pressure_delta_1d'] = grouped[pressure_col].diff(1)
    df['pressure_delta_3d'] = grouped[pressure_col].diff(3)
    print("   ✓ Pressure deltas created (weather change indicator)")

#%% 5. INTERACTION FEATURES
print("\n5. Creating interaction features...")

# Temperature-humidity interaction
if 'temperature_2m_max (°C)' in df.columns and 'relative_humidity_2m_mean (%)' in df.columns:
    df['temp_humidity_interaction'] = df['temperature_2m_max (°C)'] * df['relative_humidity_2m_mean (%)'] / 100
    print("   ✓ Temperature-humidity interaction created")

# Wind-rain interaction (wet + windy = worse air dispersion)
if 'is_rainy' in df.columns and 'wind_speed_10m_max (km/h)' in df.columns:
    df['wind_rain_interaction'] = df['is_rainy'] * df['wind_speed_10m_max (km/h)']
    print("   ✓ Wind-rain interaction created")

#%% Feature summary
print("\n" + "="*60)
print("FEATURE SUMMARY")
print("="*60)

# Get all new feature columns
new_feature_cols = [col for col in df.columns if col not in meta_cols + weather_metric_cols]

print(f"\nTotal new features created: {len(new_feature_cols)}")
print("\nFeature breakdown:")
lag_features = [col for col in new_feature_cols if '_lag_' in col]
roll_features = [col for col in new_feature_cols if 'roll' in col]
delta_features = [col for col in new_feature_cols if 'delta' in col or 'change' in col]
derived_features = [col for col in new_feature_cols if col not in lag_features + roll_features + delta_features]

print(f"  Lag features: {len(lag_features)}")
print(f"  Rolling statistics: {len(roll_features)}")
print(f"  Delta/change features: {len(delta_features)}")
print(f"  Derived metrics: {len(derived_features)}")

print("\nNULL values in new features:")
null_counts = df[new_feature_cols].isnull().sum()
null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
if len(null_counts) > 0:
    for col, count in null_counts.head(10).items():
        pct = (count / len(df)) * 100
        print(f"  {col:40s}: {count:6,} ({pct:5.2f}%)")
else:
    print("  No NULL values in new features ✓")

#%% Column ordering
print("\n" + "="*60)
print("FINALIZING DATASET")
print("="*60)

# Convert tanggal back to string for consistency
df['tanggal'] = df['tanggal'].dt.strftime('%Y-%m-%d')

# Reorder columns: meta -> original weather -> new features
final_cols = meta_cols + weather_metric_cols + new_feature_cols
df_final = df[final_cols].copy()

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Columns: {len(final_cols)} ({len(meta_cols)} meta + {len(weather_metric_cols)} original + {len(new_feature_cols)} features)")

#%% Display sample data
print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)

# Show sample with key features
sample_cols = ['tanggal', 'stasiun', 'temperature_2m_mean (°C)', 'precipitation_sum (mm)', 
               'is_rainy', 'temp_rollmean_7d', 'precip_rollsum_7d', 'temp_delta_1d']
available_sample_cols = [col for col in sample_cols if col in df_final.columns]

print("\nFirst 10 rows with selected features:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 2)
print(df_final[available_sample_cols].head(10).to_string())

print("\n\nRecent data sample (2024-2025):")
recent_data = df_final[df_final['tanggal'] >= '2024-01-01'][available_sample_cols].head(10)
print(recent_data.to_string())

#%% Statistics by station
print("\n" + "="*60)
print("STATISTICS BY STATION")
print("="*60)

if 'temperature_2m_mean (°C)' in df_final.columns and 'is_rainy' in df_final.columns:
    for station in sorted(df_final['stasiun'].unique()):
        station_data = df_final[df_final['stasiun'] == station]
        print(f"\n{station}:")
        print(f"  Mean temperature: {station_data['temperature_2m_mean (°C)'].mean():.2f}°C")
        if 'precipitation_sum (mm)' in df_final.columns:
            print(f"  Total precipitation: {station_data['precipitation_sum (mm)'].sum():.1f}mm")
        print(f"  Rainy days: {station_data['is_rainy'].sum():,} ({station_data['is_rainy'].mean()*100:.1f}%)")
        if 'heat_index' in df_final.columns:
            print(f"  Mean heat index: {station_data['heat_index'].mean():.2f}")

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
print("FEATURE CORRELATIONS (Sample)")
print("="*60)

# Select numeric columns only
numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()

# Show correlation of key features with temperature (as example)
if 'temperature_2m_mean (°C)' in numeric_cols:
    correlations = df_final[numeric_cols].corr()['temperature_2m_mean (°C)'].sort_values(ascending=False)
    print("\nTop 15 features correlated with temperature:")
    for feat, corr in correlations.head(15).items():
        if feat != 'temperature_2m_mean (°C)':
            print(f"  {feat:45s}: {corr:7.4f}")
