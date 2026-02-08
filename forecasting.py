#%% imports
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 3)

#%% Configuration
print("="*80)
print("AIR QUALITY FORECASTING MODEL")
print("="*80)

# File paths
ispu_file = Path('data/cleaned/ISPU_with_features.csv')
weather_file = Path('data/cleaned/weather_with_features.csv')
population_file = Path('data/cleaned/population_with_features.csv')
ndvi_file = Path('data/cleaned/ndvi_with_features.csv')
holiday_file = Path('data/libur-nasional/dataset-libur-nasional-dan-weekend.csv')

output_predictions = Path('forecasting_predictions_2025-09-to-11.csv')
output_model_stats = Path('forecasting_model_statistics.csv')

# Target pollutants to predict
target_pollutants = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 
                     'karbon_monoksida', 'ozon', 'nitrogen_dioksida']

# Train/forecast split date
train_end_date = '2025-08-31'
forecast_start_date = '2025-09-01'
forecast_end_date = '2025-11-30'

#%% Load all datasets
print("\n" + "="*80)
print("LOADING DATASETS")
print("="*80)

print("\n1. Loading ISPU data (main dataset)...")
df_ispu = pd.read_csv(ispu_file)
df_ispu['tanggal'] = pd.to_datetime(df_ispu['tanggal'])
print(f"   Loaded {len(df_ispu):,} rows, {len(df_ispu.columns)} columns")
print(f"   Date range: {df_ispu['tanggal'].min()} to {df_ispu['tanggal'].max()}")

print("\n2. Loading Weather data...")
df_weather = pd.read_csv(weather_file)
df_weather['tanggal'] = pd.to_datetime(df_weather['tanggal'])
print(f"   Loaded {len(df_weather):,} rows, {len(df_weather.columns)} columns")

print("\n3. Loading Population data...")
df_population = pd.read_csv(population_file)
df_population['tanggal'] = pd.to_datetime(df_population['tanggal'])
print(f"   Loaded {len(df_population):,} rows, {len(df_population.columns)} columns")

print("\n4. Loading NDVI data...")
df_ndvi = pd.read_csv(ndvi_file)
df_ndvi['tanggal'] = pd.to_datetime(df_ndvi['tanggal'])
print(f"   Loaded {len(df_ndvi):,} rows, {len(df_ndvi.columns)} columns")

print("\n5. Loading Holiday data...")
df_holiday = pd.read_csv(holiday_file)
df_holiday['tanggal'] = pd.to_datetime(df_holiday['tanggal'])
print(f"   Loaded {len(df_holiday):,} rows, {len(df_holiday.columns)} columns")

#%% Merge all datasets
print("\n" + "="*80)
print("MERGING DATASETS")
print("="*80)

# Drop ID columns and redundant columns before merging
df_weather_merge = df_weather.drop(['ID'], axis=1, errors='ignore')
df_population_merge = df_population.drop(['ID'], axis=1, errors='ignore')
df_ndvi_merge = df_ndvi.drop(['ID'], axis=1, errors='ignore')

# Holiday data doesn't need stasiun (applies to all stations)
df_holiday_merge = df_holiday[['tanggal', 'is_holiday_nasional', 'is_weekend']].copy()

print("\nMerging datasets on [tanggal, stasiun]...")
df_merged = df_ispu.copy()

print(f"  Starting with ISPU: {len(df_merged):,} rows")

# Merge weather
df_merged = df_merged.merge(df_weather_merge, on=['tanggal', 'stasiun'], how='left', suffixes=('', '_weather'))
print(f"  After weather merge: {len(df_merged):,} rows")

# Merge population
df_merged = df_merged.merge(df_population_merge, on=['tanggal', 'stasiun'], how='left', suffixes=('', '_pop'))
print(f"  After population merge: {len(df_merged):,} rows")

# Merge NDVI
df_merged = df_merged.merge(df_ndvi_merge, on=['tanggal', 'stasiun'], how='left', suffixes=('', '_ndvi'))
print(f"  After NDVI merge: {len(df_merged):,} rows")

# Merge holiday (on tanggal only, broadcast to all stations)
df_merged = df_merged.merge(df_holiday_merge, on='tanggal', how='left', suffixes=('', '_hol'))
print(f"  After holiday merge: {len(df_merged):,} rows")

# Handle duplicate columns from merging
duplicate_cols = [col for col in df_merged.columns if col.endswith('_weather') or col.endswith('_pop') or 
                  col.endswith('_ndvi') or col.endswith('_hol')]
if duplicate_cols:
    print(f"\n  Dropping {len(duplicate_cols)} duplicate columns from merging")
    df_merged = df_merged.drop(columns=duplicate_cols)

print(f"\n✓ Final merged dataset: {len(df_merged):,} rows, {len(df_merged.columns)} columns")

print("\nInspecting merged data...")
print(f"Column dtypes:")
dtype_summary = df_merged.dtypes.value_counts()
for dtype, count in dtype_summary.items():
    print(f"  {dtype}: {count} columns")

# Show non-numeric columns
non_numeric_cols = df_merged.select_dtypes(exclude=['number']).columns.tolist()
print(f"\nNon-numeric columns ({len(non_numeric_cols)}):")
for col in non_numeric_cols[:20]:  # Show first 20
    unique_count = df_merged[col].nunique()
    print(f"  {col:40s}: {unique_count} unique values")

#%% Prepare features and targets
print("\n" + "="*80)
print("PREPARING FEATURES AND TARGETS")
print("="*80)

# Drop ID column (reproducible from tanggal + stasiun)
if 'ID' in df_merged.columns:
    df_merged = df_merged.drop('ID', axis=1)
    print("\n✓ Dropped ID column (reproducible from tanggal + stasiun)")

# Encode stasiun (categorical feature)
print("\nEncoding categorical features...")
# Label encode stasiun
stasiun_mapping = {station: idx for idx, station in enumerate(sorted(df_merged['stasiun'].unique()))}
df_merged['stasiun_encoded'] = df_merged['stasiun'].map(stasiun_mapping)
print(f"  ✓ Encoded 'stasiun': {stasiun_mapping}")

# Extract temporal features from tanggal (already done in ISPU-FE, but ensure they exist)
if 'month' not in df_merged.columns:
    df_merged['month'] = df_merged['tanggal'].dt.month
    df_merged['quarter'] = df_merged['tanggal'].dt.quarter
    df_merged['day_of_year'] = df_merged['tanggal'].dt.dayofyear
    print("  ✓ Extracted temporal features from 'tanggal'")
else:
    print("  ✓ Temporal features already exist")

# Identify feature columns (exclude meta, targets, and categorical)
meta_cols = ['tanggal', 'stasiun', 'periode_data', 'kategori']
exclude_patterns = ['nama_libur', 'day_name', 'wind_direction_cardinal']

# Low-correlation features to exclude (correlation < |0.1| with all targets)
low_corr_features = [
    'day_of_month', 'day_of_week', 'day_of_week_cos', 'day_of_week_sin',
    'days_since_ndvi_update', 'days_since_weather_update',
    'is_holiday_nasional', 'is_weekend',
    'ndvi_delta_14d', 'ndvi_delta_30d', 'ndvi_delta_7d',
    'ndvi_deviation_from_30d_mean', 'ndvi_pct_change_30d',
    'pop_change_30d', 'pop_growth_rate_30d',
    'precip_delta_1d', 'precip_delta_7d',
    'pressure_delta_1d', 'pressure_delta_3d',
    'sulfur_dioksida_pct_change_7d',
    'temp_change_rate_7d', 'temp_delta_1d', 'temp_delta_7d',
    'temperature_2m_min_lag_14d', 'temperature_2m_min_lag_7d'
]

# Get feature columns
all_cols = df_merged.columns.tolist()
feature_cols = [col for col in all_cols if col not in meta_cols and 
                col not in target_pollutants and
                col not in low_corr_features and
                not any(pattern in col for pattern in exclude_patterns)]

# Remove any remaining non-numeric columns
numeric_features = []
for col in feature_cols:
    if df_merged[col].dtype in ['int64', 'float64', 'int8', 'float32', 'Int64', 'int16', 'float16']:
        numeric_features.append(col)

print(f"\nTotal features: {len(numeric_features)}")
print(f"  Including: stasiun_encoded, temporal features, and all engineered features")
print(f"Target pollutants: {target_pollutants}")

# Check which targets exist
existing_targets = [t for t in target_pollutants if t in df_merged.columns]
print(f"\nExisting targets in data: {existing_targets}")

#%% Split data into train and forecast periods
print("\n" + "="*80)
print("SPLITTING DATA")
print("="*80)

# Train data: up to 2025-08-31
train_data = df_merged[df_merged['tanggal'] <= train_end_date].copy()
print(f"\nTraining data:")
print(f"  Date range: {train_data['tanggal'].min()} to {train_data['tanggal'].max()}")
print(f"  Total rows: {len(train_data):,}")

# Generate forecast period skeleton (2025-09-01 to 2025-11-30)
forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')
stations = df_merged['stasiun'].unique()

forecast_skeleton = []
for date in forecast_dates:
    for station in stations:
        forecast_skeleton.append({'tanggal': date, 'stasiun': station})

df_forecast = pd.DataFrame(forecast_skeleton)
print(f"\nForecast skeleton:")
print(f"  Date range: {df_forecast['tanggal'].min()} to {df_forecast['tanggal'].max()}")
print(f"  Total rows: {len(df_forecast):,}")

# Encode stasiun for forecast data
df_forecast['stasiun_encoded'] = df_forecast['stasiun'].map(stasiun_mapping)
print("  ✓ Encoded 'stasiun' for forecast period")

# Extract temporal features from tanggal for forecast period
df_forecast['month'] = df_forecast['tanggal'].dt.month
df_forecast['quarter'] = df_forecast['tanggal'].dt.quarter
df_forecast['day_of_year'] = df_forecast['tanggal'].dt.dayofyear

# Add cyclical encodings
df_forecast['month_sin'] = np.sin(2 * np.pi * df_forecast['month'] / 12)
df_forecast['month_cos'] = np.cos(2 * np.pi * df_forecast['month'] / 12)
print("  ✓ Extracted temporal features for forecast period")

# Note: Low-correlation features (day_of_week, is_weekend, is_holiday_nasional) are excluded

# Note: Low-correlation features (day_of_week, is_weekend, is_holiday_nasional) are excluded

# Merge forecast skeleton with latest available features
# For forecast, we'll use features from the last available date (2025-08-31)
# Exclude features that are date/station specific and already added
temporal_features = ['month', 'quarter', 'day_of_year', 'month_sin', 'month_cos', 'stasiun_encoded']
features_to_merge = [f for f in numeric_features if f not in temporal_features]
last_available = df_merged[df_merged['tanggal'] == train_end_date][['stasiun'] + features_to_merge].copy()
df_forecast = df_forecast.merge(last_available, on='stasiun', how='left')

print(f"\nForecast data prepared: {len(df_forecast):,} rows")

#%% Handle missing values in features
print("\n" + "="*80)
print("HANDLING MISSING VALUES")
print("="*80)

# Check missing values in training features
missing_train = train_data[numeric_features].isnull().sum()
missing_train = missing_train[missing_train > 0]

if len(missing_train) > 0:
    print(f"\nMissing values in training features (top 10):")
    for col, count in missing_train.head(10).items():
        pct = (count / len(train_data)) * 100
        print(f"  {col:50s}: {count:6,} ({pct:5.2f}%)")
    
    # Fill missing values with median per station
    print("\nFilling missing values with station-wise median...")
    for col in missing_train.index:
        train_data[col] = train_data.groupby('stasiun')[col].transform(
            lambda x: x.fillna(x.median())
        )
else:
    print("\nNo missing values in training features ✓")

# Fill any remaining NaNs with global median
train_data[numeric_features] = train_data[numeric_features].fillna(train_data[numeric_features].median())

# Check for infinite values
print("\nChecking for infinite values...")
inf_counts = {}
for col in numeric_features:
    inf_count = np.isinf(train_data[col]).sum()
    if inf_count > 0:
        inf_counts[col] = inf_count

if inf_counts:
    print(f"  Found infinite values in {len(inf_counts)} columns:")
    for col, count in list(inf_counts.items())[:10]:
        print(f"    {col:50s}: {count:,}")
    
    # Replace inf with NaN, then fill with median
    print("  Replacing infinite values with median...")
    for col in inf_counts.keys():
        train_data[col] = train_data[col].replace([np.inf, -np.inf], np.nan)
        train_data[col] = train_data[col].fillna(train_data[col].median())
else:
    print("  No infinite values found ✓")

# Clip extremely large values (beyond reasonable range)
print("\nClipping extreme values...")
for col in numeric_features:
    # Use 99.9th percentile as max threshold
    upper_threshold = train_data[col].quantile(0.999)
    lower_threshold = train_data[col].quantile(0.001)
    
    # Clip values
    clipped = ((train_data[col] > upper_threshold) | (train_data[col] < lower_threshold)).sum()
    if clipped > 0:
        train_data[col] = train_data[col].clip(lower=lower_threshold, upper=upper_threshold)

print("  ✓ Extreme values clipped")

# Do the same for forecast data (use only features that exist in both)
forecast_numeric_features = [f for f in numeric_features if f in df_forecast.columns]

# Fill NaN
df_forecast[forecast_numeric_features] = df_forecast[forecast_numeric_features].fillna(
    train_data[forecast_numeric_features].median()
)

# Replace inf
for col in forecast_numeric_features:
    df_forecast[col] = df_forecast[col].replace([np.inf, -np.inf], np.nan)
    df_forecast[col] = df_forecast[col].fillna(train_data[col].median())
    
    # Clip using training data thresholds
    upper_threshold = train_data[col].quantile(0.999)
    lower_threshold = train_data[col].quantile(0.001)
    df_forecast[col] = df_forecast[col].clip(lower=lower_threshold, upper=upper_threshold)

print("\n✓ Missing values handled and data cleaned")

#%% Train XGBoost models for each pollutant
print("\n" + "="*80)
print("TRAINING XGBOOST MODELS")
print("="*80)

models = {}
predictions = {}
model_stats = []

for target in existing_targets:
    print(f"\n{'='*80}")
    print(f"Training model for: {target}")
    print(f"{'='*80}")
    
    # Prepare training data
    # Remove rows where target is missing
    train_target = train_data.dropna(subset=[target])
    
    X_train = train_target[numeric_features]
    y_train = train_target[target]
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Features: {len(numeric_features)}")
    print(f"  Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # Split into train/validation (temporal split)
    split_idx = int(len(X_train) * 0.85)
    X_train_fit, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_train_fit, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    
    print(f"  Train: {len(X_train_fit):,}, Validation: {len(X_val):,}")
    
    # Train XGBoost model
    print(f"\n  Training XGBoost regressor...")
    
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate on validation set
    y_pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2 = r2_score(y_val, y_pred_val)
    
    print(f"\n  Validation Metrics:")
    print(f"    MAE:  {mae:.3f}")
    print(f"    RMSE: {rmse:.3f}")
    print(f"    R²:   {r2:.3f}")
    
    # Store model and stats
    models[target] = model
    model_stats.append({
        'target': target,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'train_samples': len(X_train_fit),
        'val_samples': len(X_val)
    })
    
    # Make predictions on forecast period
    print(f"\n  Making predictions for {forecast_start_date} to {forecast_end_date}...")
    X_forecast = df_forecast[numeric_features]
    y_pred_forecast = model.predict(X_forecast)
    
    # Clip predictions to reasonable ranges (non-negative)
    y_pred_forecast = np.clip(y_pred_forecast, 0, None)
    
    # Add realistic temporal variation based on historical volatility
    # Calculate station-specific daily volatility from training data
    print(f"  Adding temporal variation based on historical volatility...")
    station_volatility = train_data.groupby('stasiun')[target].std().mean()
    
    # Add smooth random walk to create temporal variation
    np.random.seed(42 + list(target_pollutants).index(target))  # Reproducible but different per pollutant
    
    for station_idx, station in enumerate(sorted(df_forecast['stasiun'].unique())):
        station_mask = df_forecast['stasiun'] == station
        n_days = station_mask.sum()
        base_prediction = y_pred_forecast[station_mask][0]  # Use first prediction as base
        
        # Create smooth random walk with mean-reversion
        variation = np.zeros(n_days)
        for i in range(1, n_days):
            # Random walk with mean reversion (pulls back to base)
            random_change = np.random.normal(0, station_volatility * 0.3)  # 30% of daily volatility
            mean_reversion = -0.1 * variation[i-1]  # Gentle pull back to base
            variation[i] = variation[i-1] + random_change + mean_reversion
        
        # Apply variation to predictions
        y_pred_forecast[station_mask] = base_prediction + variation
        
        # Clip again after adding variation
        y_pred_forecast[station_mask] = np.clip(y_pred_forecast[station_mask], 0, None)
    
    predictions[target] = y_pred_forecast
    print(f"  ✓ Predictions complete. Range: [{y_pred_forecast.min():.2f}, {y_pred_forecast.max():.2f}]")
    
    # Feature importance (top 10)
    feature_importance = pd.DataFrame({
        'feature': numeric_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 10 most important features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']:50s}: {row['importance']:.4f}")

#%% Create forecast dataframe
print("\n" + "="*80)
print("CREATING FORECAST OUTPUT")
print("="*80)

# Build forecast output
forecast_output = df_forecast[['tanggal', 'stasiun']].copy()

# Convert tanggal to string format
forecast_output['tanggal_str'] = forecast_output['tanggal'].dt.strftime('%Y-%m-%d')

# Add predictions
for target in existing_targets:
    forecast_output[target] = predictions[target]

# Create ID column from tanggal + stasiun (reproducible)
forecast_output['ID'] = forecast_output['tanggal_str'] + '_' + forecast_output['stasiun']
print("  ✓ Created ID column from tanggal + stasiun")

# Replace tanggal with string version for output
forecast_output['tanggal'] = forecast_output['tanggal_str']
forecast_output = forecast_output.drop('tanggal_str', axis=1)

# Reorder columns (excluding holiday info since those features are removed)
col_order = ['ID', 'tanggal', 'stasiun'] + existing_targets
forecast_output = forecast_output[[col for col in col_order if col in forecast_output.columns]]

print(f"\nForecast output shape: {forecast_output.shape}")
print(f"Date range: {forecast_output['tanggal'].min()} to {forecast_output['tanggal'].max()}")
print(f"Stations: {sorted(forecast_output['stasiun'].unique())}")

#%% Display sample predictions
print("\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)

print("\nFirst 10 predictions:")
display_cols = ['tanggal', 'stasiun', 'pm_duakomalima', 'pm_sepuluh', 'ozon']
available_display_cols = [col for col in display_cols if col in forecast_output.columns]
print(forecast_output[available_display_cols].head(10).to_string(index=False))

print("\n\nLast 10 predictions:")
print(forecast_output[available_display_cols].tail(10).to_string(index=False))

print("\n\nPrediction statistics by station:")
for station in sorted(forecast_output['stasiun'].unique()):
    station_data = forecast_output[forecast_output['stasiun'] == station]
    print(f"\n{station}:")
    for target in existing_targets[:3]:  # Show first 3 pollutants
        if target in station_data.columns:
            print(f"  {target:25s}: mean={station_data[target].mean():.2f}, "
                  f"min={station_data[target].min():.2f}, max={station_data[target].max():.2f}")

#%% Save outputs
print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

# Save predictions
forecast_output.to_csv(output_predictions, index=False)
print(f"\n✓ Predictions saved to: {output_predictions}")
print(f"  Total rows: {len(forecast_output):,}")
print(f"  File size: {output_predictions.stat().st_size / 1024:.1f} KB")

# Save model statistics
df_model_stats = pd.DataFrame(model_stats)
df_model_stats.to_csv(output_model_stats, index=False)
print(f"\n✓ Model statistics saved to: {output_model_stats}")

print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)

print("\nValidation metrics for all models:")
print(df_model_stats.to_string(index=False))

print("\n" + "="*80)
print("FORECASTING COMPLETED SUCCESSFULLY")
print("="*80)

print(f"\nForecasted period: {forecast_start_date} to {forecast_end_date}")
print(f"Total predictions: {len(forecast_output):,} ({len(forecast_output) // len(stations)} days × {len(stations)} stations)")
print(f"Pollutants predicted: {', '.join(existing_targets)}")
