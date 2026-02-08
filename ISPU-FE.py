#%% imports
import pandas as pd
import numpy as np
from pathlib import Path

#%% Load ISPU data
print("="*60)
print("ISPU FEATURE ENGINEERING")
print("="*60)

input_file = Path('data/cleaned/ISPU_2010-2025.csv')
output_file = Path('data/cleaned/ISPU_with_features.csv')

print(f"\nLoading data from: {input_file}")
df = pd.read_csv(input_file)
df['tanggal'] = pd.to_datetime(df['tanggal'])

print(f"Loaded {len(df):,} rows")
print(f"Date range: {df['tanggal'].min()} to {df['tanggal'].max()}")
print(f"Stations: {sorted(df['stasiun'].unique())}")

#%% Sort and prepare
df = df.sort_values(['stasiun', 'tanggal']).reset_index(drop=True)

# Identify pollution columns
pollution_cols = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 
                  'ozon', 'nitrogen_dioksida']
existing_pollution_cols = [col for col in pollution_cols if col in df.columns]

print(f"\nPollution columns: {existing_pollution_cols}")

#%% Check NULL status before imputation
print("\n" + "="*60)
print("NULL VALUES BEFORE IMPUTATION")
print("="*60)

for col in existing_pollution_cols:
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df)) * 100
    print(f"  {col:25s}: {null_count:6,} ({null_pct:5.2f}%)")

#%% Fill PM2.5 based on PM10/PM2.5 ratio
print("\n" + "="*60)
print("FILLING PM2.5 BASED ON PM10 RATIO")
print("="*60)

if 'pm_duakomalima' in df.columns and 'pm_sepuluh' in df.columns:
    # Calculate PM10/PM2.5 ratio from known data
    valid_data = df[(df['pm_duakomalima'].notna()) & (df['pm_sepuluh'].notna()) & 
                    (df['pm_duakomalima'] > 0)].copy()
    
    if len(valid_data) > 0:
        valid_data['pm_ratio'] = valid_data['pm_sepuluh'] / valid_data['pm_duakomalima']
        
        # Calculate average ratio per station (more accurate)
        station_ratios = valid_data.groupby('stasiun')['pm_ratio'].median()
        overall_ratio = valid_data['pm_ratio'].median()
        
        print(f"\nPM10/PM2.5 ratio statistics:")
        print(f"  Overall median ratio: {overall_ratio:.3f}")
        print(f"  Ratio by station:")
        for station, ratio in station_ratios.items():
            print(f"    {station}: {ratio:.3f}")
        
        # Fill missing PM2.5 values using station-specific ratios
        missing_pm25 = df['pm_duakomalima'].isna() & df['pm_sepuluh'].notna()
        n_missing = missing_pm25.sum()
        
        if n_missing > 0:
            print(f"\n  Found {n_missing:,} rows with PM10 but missing PM2.5")
            
            for station in df['stasiun'].unique():
                station_mask = (df['stasiun'] == station) & missing_pm25
                if station_mask.sum() > 0:
                    # Use station-specific ratio if available, else overall
                    ratio = station_ratios.get(station, overall_ratio)
                    df.loc[station_mask, 'pm_duakomalima'] = df.loc[station_mask, 'pm_sepuluh'] / ratio
                    print(f"    {station}: Filled {station_mask.sum():,} values using ratio {ratio:.3f}")
        else:
            print(f"  No missing PM2.5 values to fill")
    else:
        print("  No valid PM10/PM2.5 pairs found in data")
else:
    print("  PM10 or PM2.5 column not found")

#%% Forward fill NULL values per station
print("\n" + "="*60)
print("FORWARD FILLING NULL VALUES")
print("="*60)

grouped = df.groupby('stasiun', group_keys=False)

for col in existing_pollution_cols:
    before_null = df[col].isnull().sum()
    df[col] = grouped[col].ffill()
    after_null = df[col].isnull().sum()
    filled = before_null - after_null
    if filled > 0:
        print(f"  {col:25s}: Filled {filled:,} values")

print("\nRemaining NULL values after forward fill:")
for col in existing_pollution_cols:
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df)) * 100
    print(f"  {col:25s}: {null_count:6,} ({null_pct:5.2f}%)")

#%% Feature Engineering
print("\n" + "="*60)
print("CREATING FEATURES")
print("="*60)

#%% 1. LAG FEATURES
print("\n1. Creating lag features (7, 14, 30 days)...")

lag_periods = [7, 14, 30]
for col in existing_pollution_cols:
    for lag in lag_periods:
        df[f'{col}_lag_{lag}d'] = grouped[col].shift(lag)

print(f"   ✓ Created lag features for {len(existing_pollution_cols)} pollutants")

#%% 2. ROLLING STATISTICS
print("\n2. Creating rolling statistics (7, 14, 30 days)...")

for col in existing_pollution_cols:
    # Rolling means
    df[f'{col}_rollmean_7d'] = grouped[col].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df[f'{col}_rollmean_14d'] = grouped[col].transform(lambda x: x.rolling(14, min_periods=1).mean())
    df[f'{col}_rollmean_30d'] = grouped[col].transform(lambda x: x.rolling(30, min_periods=1).mean())
    
    # Rolling std (volatility)
    df[f'{col}_rollstd_30d'] = grouped[col].transform(lambda x: x.rolling(30, min_periods=7).std())

print(f"   ✓ Created rolling statistics for {len(existing_pollution_cols)} pollutants")

#%% 3. DELTAS AND CHANGES
print("\n3. Creating delta and change features...")

for col in existing_pollution_cols:
    df[f'{col}_delta_1d'] = grouped[col].diff(1)
    df[f'{col}_delta_7d'] = grouped[col].diff(7)
    df[f'{col}_delta_30d'] = grouped[col].diff(30)
    df[f'{col}_pct_change_7d'] = grouped[col].pct_change(7)

print(f"   ✓ Created delta features for {len(existing_pollution_cols)} pollutants")

#%% 4. DERIVED METRICS
print("\n4. Creating derived air quality metrics...")

# PM2.5/PM10 ratio (fine particle concentration)
if 'pm_duakomalima' in df.columns and 'pm_sepuluh' in df.columns:
    df['pm_fine_ratio'] = df['pm_duakomalima'] / df['pm_sepuluh'].replace(0, np.nan)
    print("   ✓ PM fine ratio (PM2.5/PM10) created")

# Air Quality Index (AQI) approximation
# Simple weighted sum of normalized pollutants
if all(col in df.columns for col in ['pm_sepuluh', 'pm_duakomalima', 'ozon']):
    # Normalize key pollutants (rough approximation)
    df['aqi_proxy'] = (
        df['pm_duakomalima'] * 0.5 +  # PM2.5 most important
        df['pm_sepuluh'] * 0.3 +       # PM10
        df['ozon'] * 0.2               # Ozone
    )
    print("   ✓ AQI proxy created (weighted pollutant sum)")

# Pollution spike indicators (above 90th percentile)
for col in existing_pollution_cols:
    threshold = df[col].quantile(0.90)
    df[f'{col}_spike'] = (df[col] > threshold).astype('int8')

print(f"   ✓ Pollution spike indicators created (90th percentile threshold)")

# Multi-pollutant count (how many pollutants are elevated)
if 'pm_sepuluh_spike' in df.columns:
    spike_cols = [f'{col}_spike' for col in existing_pollution_cols]
    df['elevated_pollutant_count'] = df[spike_cols].sum(axis=1)
    print("   ✓ Elevated pollutant count created")

# Unhealthy air flag (kategori based)
if 'kategori' in df.columns:
    df['is_unhealthy'] = df['kategori'].isin(['TIDAK SEHAT', 'SANGAT TIDAK SEHAT', 'BERBAHAYA']).astype('int8')
    
    # Count unhealthy days in rolling windows
    df['unhealthy_days_7d'] = grouped['is_unhealthy'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df['unhealthy_days_30d'] = grouped['is_unhealthy'].transform(lambda x: x.rolling(30, min_periods=1).sum())
    print("   ✓ Unhealthy air indicators created")

#%% 5. INTERACTION FEATURES
print("\n5. Creating interaction features...")

# PM interactions
if 'pm_duakomalima' in df.columns and 'pm_sepuluh' in df.columns:
    df['pm_total'] = df['pm_duakomalima'] + df['pm_sepuluh']
    print("   ✓ Total PM (PM2.5 + PM10) created")

# Gaseous pollutant index
if all(col in df.columns for col in ['sulfur_dioksida', 'karbon_monoksida', 'nitrogen_dioksida']):
    df['gaseous_pollutant_index'] = (
        df['sulfur_dioksida'] + 
        df['karbon_monoksida'] / 100 +  # CO is typically much higher, scale down
        df['nitrogen_dioksida']
    )
    print("   ✓ Gaseous pollutant index created")

#%% 6. TEMPORAL FEATURES
print("\n6. Creating temporal features...")

df['day_of_week'] = df['tanggal'].dt.dayofweek
df['day_of_month'] = df['tanggal'].dt.day
df['month'] = df['tanggal'].dt.month
df['quarter'] = df['tanggal'].dt.quarter
df['day_of_year'] = df['tanggal'].dt.dayofyear

# Weekend flag
df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')

# Cyclical encoding for day of week and month
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print("   ✓ Temporal features created (day, week, month, cyclical encodings)")

#%% 7. DEVIATION FROM MEAN
print("\n7. Creating deviation features...")

for col in existing_pollution_cols:
    rollmean_col = f'{col}_rollmean_30d'
    if rollmean_col in df.columns:
        df[f'{col}_deviation_from_30d_mean'] = df[col] - df[rollmean_col]
        df[f'{col}_deviation_pct'] = ((df[col] - df[rollmean_col]) / df[rollmean_col].replace(0, np.nan)) * 100

print(f"   ✓ Deviation features created for {len(existing_pollution_cols)} pollutants")

#%% Feature summary
print("\n" + "="*60)
print("FEATURE SUMMARY")
print("="*60)

# Get all columns
meta_cols = ['ID', 'periode_data', 'tanggal', 'stasiun']
original_cols = existing_pollution_cols + ['kategori']
new_feature_cols = [col for col in df.columns if col not in meta_cols + original_cols]

print(f"\nTotal new features created: {len(new_feature_cols)}")
print("\nFeature breakdown:")
lag_features = [col for col in new_feature_cols if '_lag_' in col]
roll_features = [col for col in new_feature_cols if 'roll' in col]
delta_features = [col for col in new_feature_cols if 'delta' in col or 'pct_change' in col]
spike_features = [col for col in new_feature_cols if 'spike' in col or 'unhealthy' in col or 'elevated' in col]
temporal_features = [col for col in new_feature_cols if any(x in col for x in ['day', 'week', 'month', 'quarter', 'year', 'sin', 'cos'])]
interaction_features = [col for col in new_feature_cols if any(x in col for x in ['ratio', 'total', 'index', 'proxy', 'aqi'])]
deviation_features = [col for col in new_feature_cols if 'deviation' in col]

print(f"  Lag features: {len(lag_features)}")
print(f"  Rolling statistics: {len(roll_features)}")
print(f"  Delta/change features: {len(delta_features)}")
print(f"  Spike/threshold indicators: {len(spike_features)}")
print(f"  Temporal features: {len(temporal_features)}")
print(f"  Interaction features: {len(interaction_features)}")
print(f"  Deviation features: {len(deviation_features)}")

print("\nNULL values in new features (top 10):")
null_counts = df[new_feature_cols].isnull().sum()
null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
if len(null_counts) > 0:
    for col, count in null_counts.head(10).items():
        pct = (count / len(df)) * 100
        print(f"  {col:45s}: {count:6,} ({pct:5.2f}%)")
else:
    print("  No NULL values in new features ✓")

#%% Column ordering
print("\n" + "="*60)
print("FINALIZING DATASET")
print("="*60)

# Convert tanggal back to string for consistency
df['tanggal'] = df['tanggal'].dt.strftime('%Y-%m-%d')

# Reorder columns: meta -> original -> new features
final_cols = meta_cols + original_cols + new_feature_cols
df_final = df[final_cols].copy()

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Columns: {len(final_cols)} ({len(meta_cols)} meta + {len(original_cols)} original + {len(new_feature_cols)} features)")

#%% Display sample data
print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)

sample_cols = ['tanggal', 'stasiun', 'pm_duakomalima', 'pm_sepuluh', 'kategori',
               'pm_duakomalima_rollmean_7d', 'pm_duakomalima_delta_7d', 'is_unhealthy']
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

for station in sorted(df_final['stasiun'].unique()):
    station_data = df_final[df_final['stasiun'] == station]
    print(f"\n{station}:")
    if 'pm_duakomalima' in df_final.columns:
        print(f"  Mean PM2.5: {station_data['pm_duakomalima'].mean():.2f}")
        print(f"  Max PM2.5: {station_data['pm_duakomalima'].max():.2f}")
    if 'is_unhealthy' in df_final.columns:
        unhealthy_count = station_data['is_unhealthy'].sum()
        unhealthy_pct = (unhealthy_count / len(station_data)) * 100
        print(f"  Unhealthy days: {unhealthy_count:,} ({unhealthy_pct:.1f}%)")

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

# Show correlation of key features with PM2.5 (if available)
if 'pm_duakomalima' in numeric_cols:
    correlations = df_final[numeric_cols].corr()['pm_duakomalima'].sort_values(ascending=False)
    print("\nTop 15 features correlated with PM2.5:")
    for feat, corr in correlations.head(15).items():
        if feat != 'pm_duakomalima':
            print(f"  {feat:50s}: {corr:7.4f}")
