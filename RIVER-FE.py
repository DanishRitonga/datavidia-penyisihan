# %% [markdown]
# # River Water Quality - Feature Engineering
# ## Preparing River Data for Air Quality Forecasting
# 
# This script engineers features from river water quality data to predict air pollutants.
# 
# **Key Steps:**
# 1. Load daily river quality data and ISPU (air quality) data
# 2. Create temporal features (lags, rolling stats, changes)
# 3. Create seasonal/cyclical features
# 4. Analyze correlations with air pollutants
# 5. Select most relevant parameters
# 6. Create station-specific features
# 7. Export model-ready dataset

# %% [markdown]
## 1. Setup & Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# %% [markdown]
## 2. Configuration

# %%
# File paths
RIVER_DATA_PATH = 'data/cleaned/river_quality_2015-2025_daily.csv'
ISPU_DATA_PATH = 'data/cleaned/ISPU_2010-2025.csv'
OUTPUT_PATH = 'data/cleaned/river_quality_features.csv'

# Feature engineering parameters
LAG_PERIODS = [7, 14, 30, 90, 180]  # days
ROLLING_WINDOWS = [7, 30, 90]  # days
CHANGE_PERIOD = 30  # days for change calculation

# Station codes
STATIONS = ['DKI1', 'DKI2', 'DKI3', 'DKI4', 'DKI5']

# Base water quality parameters (before feature engineering)
BASE_PARAMS = [
    'biological_oxygen_demand',
    'cadmium',
    'chemical_oxygen_demand',
    'chromium_vi',
    'copper',
    'fecal_coliform',
    'lead',
    'mbas_detergent',
    'mercury',
    'oil_and_grease',
    'ph',
    'total_coliform',
    'total_dissolved_solids',
    'total_suspended_solids',
    'zinc'
]

# Air pollutants to analyze correlation with
AIR_POLLUTANTS = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 
                  'karbon_monoksida', 'ozon', 'nitrogen_dioksida']

# Air quality category encoding (ordinal)
KATEGORI_ENCODING = {
    'BAIK': 1,
    'SEDANG': 2,
    'TIDAK SEHAT': 3,
    'SANGAT TIDAK SEHAT': 4,
    'BERBAHAYA': 5
}

# %% [markdown]
## 3. Load Data

# %%
print("="*70)
print("LOADING DATA")
print("="*70)

# Load river quality data
print("\n[1] Loading river quality data...")
df_river = pd.read_csv(RIVER_DATA_PATH)
df_river['tanggal'] = pd.to_datetime(df_river['tanggal'])
print(f"  ✓ Loaded: {df_river.shape}")
print(f"  → Date range: {df_river['tanggal'].min().date()} to {df_river['tanggal'].max().date()}")

# Load ISPU (air quality) data
print("\n[2] Loading ISPU data...")
df_ispu = pd.read_csv(ISPU_DATA_PATH)
df_ispu['tanggal'] = pd.to_datetime(df_ispu['tanggal'])

# Encode kategori column (ordinal encoding)
if 'kategori' in df_ispu.columns:
    df_ispu['kategori_encoded'] = df_ispu['kategori'].map(KATEGORI_ENCODING)
    print(f"  ✓ Loaded: {df_ispu.shape}")
    print(f"  → Date range: {df_ispu['tanggal'].min().date()} to {df_ispu['tanggal'].max().date()}")
    print(f"  → Encoded kategori: {df_ispu['kategori'].unique().tolist()}")
    print(f"    Encoding: BAIK=1, SEDANG=2, TIDAK SEHAT=3, SANGAT TIDAK SEHAT=4, BERBAHAYA=5")
else:
    print(f"  ✓ Loaded: {df_ispu.shape}")
    print(f"  → Date range: {df_ispu['tanggal'].min().date()} to {df_ispu['tanggal'].max().date()}")

# Find overlapping date range
overlap_start = max(df_river['tanggal'].min(), df_ispu['tanggal'].min())
overlap_end = min(df_river['tanggal'].max(), df_ispu['tanggal'].max())
print(f"\n  → Overlapping period: {overlap_start.date()} to {overlap_end.date()}")
print(f"  → Days overlap: {(overlap_end - overlap_start).days + 1}")

# %% [markdown]
## 4. Basic Data Validation

# %%
print("\n" + "="*70)
print("DATA VALIDATION")
print("="*70)

print("\n[1] River data summary:")
print(f"  - Stations: {sorted(df_river['stasiun'].unique())}")
print(f"  - Records per station: {df_river.groupby('stasiun').size().to_dict()}")
print(f"  - Parameters: {len(BASE_PARAMS)}")

print("\n[2] ISPU data summary:")
print(f"  - Stations: {sorted(df_ispu['stasiun'].unique())}")
print(f"  - Air pollutants: {len(AIR_POLLUTANTS)}")
print(f"  - Missing values:")
for pollutant in AIR_POLLUTANTS:
    missing_pct = (df_ispu[pollutant].isna().sum() / len(df_ispu) * 100)
    print(f"     {pollutant}: {missing_pct:.1f}%")
if 'kategori_encoded' in df_ispu.columns:
    missing_pct = (df_ispu['kategori_encoded'].isna().sum() / len(df_ispu) * 100)
    print(f"     kategori_encoded: {missing_pct:.1f}%")

# %% [markdown]
## 5. Create Temporal Features

# %% [markdown]
### 5.1 Lag Features

# %%
print("\n" + "="*70)
print("CREATING TEMPORAL FEATURES")
print("="*70)

print("\n[1] Creating lag features...")

# Initialize feature dataframe
df_features = df_river[['tanggal', 'stasiun']].copy()

# Add base parameters
for param in BASE_PARAMS:
    if param in df_river.columns:
        df_features[param] = df_river[param]

# Create lag features for each parameter and station
lag_features_created = 0

for station in STATIONS:
    station_mask = df_features['stasiun'] == station
    
    for param in BASE_PARAMS:
        if param not in df_features.columns:
            continue
            
        station_data = df_features.loc[station_mask].copy()
        station_data = station_data.sort_values('tanggal').reset_index(drop=True)
        
        for lag in LAG_PERIODS:
            lag_col = f'{param}_lag_{lag}d'
            lag_values = station_data[param].shift(lag)
            df_features.loc[station_mask, lag_col] = lag_values.values
            lag_features_created += 1

print(f"  ✓ Created {lag_features_created} lag features")
print(f"    Lags: {LAG_PERIODS} days")

# %% [markdown]
### 5.2 Rolling Statistics

# %%
print("\n[2] Creating rolling statistics...")

rolling_features_created = 0

for station in STATIONS:
    station_mask = df_features['stasiun'] == station
    
    for param in BASE_PARAMS:
        if param not in df_features.columns:
            continue
            
        station_data = df_features.loc[station_mask].copy()
        station_data = station_data.sort_values('tanggal').reset_index(drop=True)
        
        for window in ROLLING_WINDOWS:
            # Rolling mean
            mean_col = f'{param}_roll_mean_{window}d'
            mean_values = station_data[param].rolling(window=window, min_periods=1).mean()
            df_features.loc[station_mask, mean_col] = mean_values.values
            rolling_features_created += 1
            
            # Rolling std
            std_col = f'{param}_roll_std_{window}d'
            std_values = station_data[param].rolling(window=window, min_periods=1).std()
            df_features.loc[station_mask, std_col] = std_values.values
            rolling_features_created += 1

print(f"  ✓ Created {rolling_features_created} rolling features")
print(f"    Windows: {ROLLING_WINDOWS} days (mean + std)")

# %% [markdown]
### 5.3 Change Features

# %%
print("\n[3] Creating change features...")

change_features_created = 0

for station in STATIONS:
    station_mask = df_features['stasiun'] == station
    
    for param in BASE_PARAMS:
        if param not in df_features.columns:
            continue
            
        station_data = df_features.loc[station_mask].copy()
        station_data = station_data.sort_values('tanggal').reset_index(drop=True)
        
        # Absolute change (current - 30d ago)
        change_col = f'{param}_change_{CHANGE_PERIOD}d'
        lagged = station_data[param].shift(CHANGE_PERIOD)
        change_values = station_data[param] - lagged
        df_features.loc[station_mask, change_col] = change_values.values
        change_features_created += 1
        
        # Percent change
        pct_change_col = f'{param}_pct_change_{CHANGE_PERIOD}d'
        pct_change_values = ((station_data[param] - lagged) / (lagged + 1e-8)) * 100
        df_features.loc[station_mask, pct_change_col] = pct_change_values.values
        change_features_created += 1

print(f"  ✓ Created {change_features_created} change features")
print(f"    Period: {CHANGE_PERIOD} days (absolute + percent)")

# %% [markdown]
## 6. Create Seasonal & Temporal Features

# %%
print("\n" + "="*70)
print("CREATING SEASONAL FEATURES")
print("="*70)

# Extract datetime components
df_features['year'] = df_features['tanggal'].dt.year
df_features['month'] = df_features['tanggal'].dt.month
df_features['quarter'] = df_features['tanggal'].dt.quarter
df_features['day_of_year'] = df_features['tanggal'].dt.dayofyear
df_features['week_of_year'] = df_features['tanggal'].dt.isocalendar().week

# Wet season in Jakarta (November - March)
df_features['is_wet_season'] = df_features['month'].isin([11, 12, 1, 2, 3]).astype(int)

# Yearly trend (years since 2015)
df_features['yearly_trend'] = df_features['year'] - 2015

# Cyclical encoding for month (preserves circular nature)
df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)

# Cyclical encoding for day of year
df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365.25)
df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365.25)

seasonal_features = ['year', 'month', 'quarter', 'day_of_year', 'week_of_year',
                     'is_wet_season', 'yearly_trend', 'month_sin', 'month_cos',
                     'day_sin', 'day_cos']

print(f"  ✓ Created {len(seasonal_features)} seasonal/temporal features:")
for feat in seasonal_features:
    print(f"     - {feat}")

# %% [markdown]
## 7. Correlation Analysis with Air Pollutants

# %%
print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

# Merge river features with ISPU data for correlation analysis
print("\n[1] Merging datasets for correlation analysis...")
merge_cols = ['tanggal', 'stasiun'] + AIR_POLLUTANTS
if 'kategori_encoded' in df_ispu.columns:
    merge_cols.append('kategori_encoded')
df_merged = df_features.merge(
    df_ispu[merge_cols],
    on=['tanggal', 'stasiun'],
    how='inner'
)
print(f"  ✓ Merged shape: {df_merged.shape}")
print(f"  → Common records: {len(df_merged):,}")

# %% [markdown]
### 7.1 Base Parameter Correlations

# %%
print("\n[2] Analyzing base parameter correlations...")

# Calculate correlations for base parameters only
correlation_results = []

# Define targets for correlation (air pollutants + kategori)
targets_for_corr = AIR_POLLUTANTS.copy()
if 'kategori_encoded' in df_merged.columns:
    targets_for_corr.append('kategori_encoded')

for param in BASE_PARAMS:
    if param not in df_merged.columns:
        continue
    
    param_corr = {}
    param_corr['parameter'] = param
    
    for pollutant in targets_for_corr:
        # Calculate correlation (handle NaNs)
        valid_mask = df_merged[param].notna() & df_merged[pollutant].notna()
        if valid_mask.sum() > 30:  # Need at least 30 valid pairs
            corr = df_merged.loc[valid_mask, param].corr(df_merged.loc[valid_mask, pollutant])
            param_corr[pollutant] = corr
        else:
            param_corr[pollutant] = np.nan
    
    # Average absolute correlation
    abs_corrs = [abs(v) for v in param_corr.values() if isinstance(v, float) and not np.isnan(v)]
    param_corr['avg_abs_corr'] = np.mean(abs_corrs) if abs_corrs else 0
    
    correlation_results.append(param_corr)

df_correlations = pd.DataFrame(correlation_results)
df_correlations = df_correlations.sort_values('avg_abs_corr', ascending=False)

print("\n  Correlation with air pollutants (sorted by average absolute correlation):")
print("  " + "-"*66)
pd.set_option('display.float_format', lambda x: f'{x:+.3f}' if not pd.isna(x) else 'NaN')
print(df_correlations.to_string(index=False))
pd.set_option('display.float_format', None)

# Show top correlations with kategori_encoded specifically
if 'kategori_encoded' in df_correlations.columns:
    print("\n  Top 10 parameters correlated with Air Quality Category (kategori_encoded):")
    kategori_corr = df_correlations[['parameter', 'kategori_encoded', 'avg_abs_corr']].copy()
    kategori_corr = kategori_corr.sort_values('kategori_encoded', key=abs, ascending=False)
    print("  " + "-"*66)
    for idx, row in kategori_corr.head(10).iterrows():
        param = row['parameter']
        corr = row['kategori_encoded']
        if not pd.isna(corr):
            print(f"    {param:<40} r={corr:+.3f}")

# %% [markdown]
### 7.2 Select Top Parameters

# %%
print("\n[3] Selecting top parameters...")

# Define priority based on correlation and domain knowledge
TOP_N_PARAMS = 10

# Get top correlated parameters
top_params_by_corr = df_correlations.nlargest(TOP_N_PARAMS, 'avg_abs_corr')['parameter'].tolist()

# Define must-have parameters (domain knowledge)
must_have_params = [
    'biological_oxygen_demand',      # BOD - organic pollution indicator
    'chemical_oxygen_demand',         # COD - organic pollution indicator
    'fecal_coliform',                 # Bacterial contamination
    'total_coliform',                 # Bacterial contamination
    'total_suspended_solids',         # Particulate matter analog
    'ph',                             # Acidity/alkalinity
]

# Combine and deduplicate
priority_params = list(dict.fromkeys(must_have_params + top_params_by_corr))[:TOP_N_PARAMS]

print(f"\n  ✓ Selected {len(priority_params)} priority parameters:")
for i, param in enumerate(priority_params, 1):
    avg_corr = df_correlations[df_correlations['parameter'] == param]['avg_abs_corr'].values[0]
    print(f"     {i:2d}. {param:<35} (avg |r| = {avg_corr:.3f})")

print(f"\n  Parameters excluded: {set(BASE_PARAMS) - set(priority_params)}")

# %% [markdown]
## 8. Filter Features by Priority Parameters

# %%
print("\n" + "="*70)
print("FILTERING FEATURES")
print("="*70)

# Identify all feature columns related to priority parameters
priority_feature_cols = ['tanggal', 'stasiun']

# Add seasonal features (always keep)
priority_feature_cols.extend(seasonal_features)

# Add base parameters, lags, rolling, and change features for priority parameters only
for param in priority_params:
    # Base parameter
    if param in df_features.columns:
        priority_feature_cols.append(param)
    
    # Lag features
    for lag in LAG_PERIODS:
        lag_col = f'{param}_lag_{lag}d'
        if lag_col in df_features.columns:
            priority_feature_cols.append(lag_col)
    
    # Rolling features
    for window in ROLLING_WINDOWS:
        mean_col = f'{param}_roll_mean_{window}d'
        std_col = f'{param}_roll_std_{window}d'
        if mean_col in df_features.columns:
            priority_feature_cols.append(mean_col)
        if std_col in df_features.columns:
            priority_feature_cols.append(std_col)
    
    # Change features
    change_col = f'{param}_change_{CHANGE_PERIOD}d'
    pct_change_col = f'{param}_pct_change_{CHANGE_PERIOD}d'
    if change_col in df_features.columns:
        priority_feature_cols.append(change_col)
    if pct_change_col in df_features.columns:
        priority_feature_cols.append(pct_change_col)

# Filter dataframe
df_features_filtered = df_features[priority_feature_cols].copy()

print(f"\n  Original features: {len(df_features.columns)}")
print(f"  Filtered features: {len(df_features_filtered.columns)}")
print(f"  Reduction: {len(df_features.columns) - len(df_features_filtered.columns)} columns")

# %% [markdown]
## 9. Create Station-Specific Features

# %%
print("\n" + "="*70)
print("CREATING STATION FEATURES")
print("="*70)

# One-hot encoding for stations
print("\n[1] Creating one-hot encoded station features...")
station_dummies = pd.get_dummies(df_features_filtered['stasiun'], prefix='station')
df_features_filtered = pd.concat([df_features_filtered, station_dummies], axis=1)

print(f"  ✓ Created {len(station_dummies.columns)} station dummy features:")
for col in station_dummies.columns:
    print(f"     - {col}")

# %% [markdown]
### 9.1 Target Encoding (Mean Encoding)

# %%
print("\n[2] Creating target encoding for stations...")

# Merge with ISPU to get air pollutant targets
target_cols = ['tanggal', 'stasiun'] + AIR_POLLUTANTS
if 'kategori_encoded' in df_ispu.columns:
    target_cols.append('kategori_encoded')
df_with_targets = df_features_filtered.merge(
    df_ispu[target_cols],
    on=['tanggal', 'stasiun'],
    how='left'
)

# Calculate mean pollutant levels per station (target encoding)
target_encode_cols = AIR_POLLUTANTS.copy()
if 'kategori_encoded' in df_with_targets.columns:
    target_encode_cols.append('kategori_encoded')

for pollutant in target_encode_cols:
    station_means = df_with_targets.groupby('stasiun')[pollutant].mean()
    encoding_col = f'station_mean_{pollutant}'
    df_features_filtered[encoding_col] = df_features_filtered['stasiun'].map(station_means)

print(f"  ✓ Created {len(target_encode_cols)} target-encoded station features")

# Show station means
print("\n  Station mean air pollutant levels:")
station_summary = df_with_targets.groupby('stasiun')[AIR_POLLUTANTS].mean()
print(station_summary.round(2))

if 'kategori_encoded' in df_with_targets.columns:
    print("\n  Station mean air quality category (1=BAIK, 2=SEDANG, 3=TIDAK SEHAT, etc.):")
    kategori_summary = df_with_targets.groupby('stasiun')['kategori_encoded'].mean()
    print(kategori_summary.round(2))

# %% [markdown]
## 10. Feature Summary

# %%
print("\n" + "="*70)
print("FEATURE SUMMARY")
print("="*70)

# Categorize features
feature_categories = {
    'Metadata': ['tanggal', 'stasiun'],
    'Base Parameters': [col for col in df_features_filtered.columns 
                        if col in priority_params],
    'Lag Features': [col for col in df_features_filtered.columns 
                     if '_lag_' in col],
    'Rolling Features': [col for col in df_features_filtered.columns 
                         if '_roll_' in col],
    'Change Features': [col for col in df_features_filtered.columns 
                        if '_change_' in col or '_pct_change_' in col],
    'Seasonal Features': [col for col in df_features_filtered.columns 
                          if col in seasonal_features],
    'Station Features': [col for col in df_features_filtered.columns 
                         if col.startswith('station_')],
}

print("\nFeature breakdown by category:")
total_features = 0
for category, features in feature_categories.items():
    count = len(features)
    total_features += count
    print(f"  {category:<20} {count:>4} features")

print(f"  {'-'*28}")
print(f"  {'TOTAL':<20} {total_features:>4} features")

# %% [markdown]
## 11. Data Quality Check

# %%
print("\n" + "="*70)
print("DATA QUALITY CHECK")
print("="*70)

# Missing values analysis
print("\n[1] Missing values in engineered features:")
missing_counts = df_features_filtered.isnull().sum()
missing_pct = (missing_counts / len(df_features_filtered) * 100)

# Show features with >5% missing
high_missing = missing_pct[missing_pct > 5].sort_values(ascending=False)
if len(high_missing) > 0:
    print(f"\n  Features with >5% missing:")
    for feat, pct in high_missing.head(20).items():
        print(f"    {feat:<50} {pct:>6.2f}%")
else:
    print("  ✓ No features with >5% missing values")

# Check for infinite values
print("\n[2] Checking for infinite values...")
numeric_cols = df_features_filtered.select_dtypes(include=[np.number]).columns
inf_counts = {}
for col in numeric_cols:
    inf_count = np.isinf(df_features_filtered[col]).sum()
    if inf_count > 0:
        inf_counts[col] = inf_count

if inf_counts:
    print(f"  ⚠ Found infinite values:")
    for col, count in sorted(inf_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {col:<50} {count:>6} inf values")
    
    # Replace inf with NaN
    print("\n  → Replacing infinite values with NaN...")
    df_features_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)
else:
    print("  ✓ No infinite values found")

# %% [markdown]
## 12. Export Features

# %%
print("\n" + "="*70)
print("EXPORTING FEATURES")
print("="*70)

# Sort by date and station
df_features_filtered = df_features_filtered.sort_values(['tanggal', 'stasiun']).reset_index(drop=True)

# Export
df_features_filtered.to_csv(OUTPUT_PATH, index=False)

print(f"\n  ✓ Saved to: {OUTPUT_PATH}")
print(f"  → Shape: {df_features_filtered.shape}")
print(f"  → Date range: {df_features_filtered['tanggal'].min().date()} to {df_features_filtered['tanggal'].max().date()}")
print(f"  → Total features: {len(df_features_filtered.columns)}")
print(f"  → File size: {Path(OUTPUT_PATH).stat().st_size / 1024 / 1024:.2f} MB")

# %% [markdown]
## 13. Feature Importance Preview

# %%
print("\n" + "="*70)
print("FEATURE IMPORTANCE PREVIEW")
print("="*70)

# Show top features by variance (proxy for importance)
print("\n[1] Top features by variance (excluding metadata):")
numeric_features = df_features_filtered.select_dtypes(include=[np.number]).columns
feature_variance = df_features_filtered[numeric_features].var().sort_values(ascending=False)

print("\n  Top 20 most variable features:")
for i, (feat, var) in enumerate(feature_variance.head(20).items(), 1):
    print(f"    {i:2d}. {feat:<50} variance={var:.2e}")

# %% [markdown]
## 14. Final Summary

# %%
print("\n" + "="*70)
print("FEATURE ENGINEERING COMPLETE")
print("="*70)

print(f"\nInput:")
print(f"  - River data: {df_river.shape}")
print(f"  - ISPU data: {df_ispu.shape}")

print(f"\nOutput:")
print(f"  - Engineered features: {df_features_filtered.shape}")
print(f"  - File: {OUTPUT_PATH}")

print(f"\nFeature engineering applied:")
print(f"  ✓ Temporal lags: {LAG_PERIODS}")
print(f"  ✓ Rolling windows: {ROLLING_WINDOWS}")
print(f"  ✓ Change period: {CHANGE_PERIOD} days")
print(f"  ✓ Seasonal features: {len(seasonal_features)}")
print(f"  ✓ Station encoding: one-hot + target encoding")
print(f"  ✓ Parameter selection: {len(priority_params)} of {len(BASE_PARAMS)} parameters")

print(f"\nTop priority parameters:")
for i, param in enumerate(priority_params[:5], 1):
    print(f"  {i}. {param}")

print(f"\nRecommendations for modeling:")
print(f"  → Use features with low missing % and high correlation")
print(f"  → Consider staleness indicators from original data")
print(f"  → Apply feature scaling (StandardScaler or MinMaxScaler)")
print(f"  → Use CV with temporal splits to avoid look-ahead bias")
print(f"  → Consider model ensembles (XGBoost, LightGBM, CatBoost)")

print("="*70)
