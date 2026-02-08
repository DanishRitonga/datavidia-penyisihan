#%% imports
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

pd.set_option('display.max_columns', None)

#%% Configuration
print("="*80)
print("AIR QUALITY CATEGORY CLASSIFICATION MODEL")
print("="*80)

# File paths
ispu_file = Path('data/cleaned/ISPU_with_features.csv')
weather_file = Path('data/cleaned/weather_with_features.csv')
population_file = Path('data/cleaned/population_with_features.csv')
ndvi_file = Path('data/cleaned/ndvi_with_features.csv')
holiday_file = Path('data/libur-nasional/dataset-libur-nasional-dan-weekend.csv')

# Load predicted pollutant values for forecast period
pollutant_predictions_file = Path('forecasting_predictions_2025-09-to-11.csv')

output_with_kategori = Path('forecasting_predictions_with_kategori_2025-09-to-11.csv')
submission_file = Path('submission.csv')
model_stats_file = Path('classification_model_statistics.csv')

# Train/forecast split date
train_end_date = '2025-08-31'
forecast_start_date = '2025-09-01'
forecast_end_date = '2025-11-30'

# Train/forecast split date
train_end_date = '2025-08-31'
forecast_start_date = '2025-09-01'
forecast_end_date = '2025-11-30'

#%% Load all datasets (same as forecasting.py)
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

# Merge holiday
df_merged = df_merged.merge(df_holiday_merge, on='tanggal', how='left', suffixes=('', '_hol'))
print(f"  After holiday merge: {len(df_merged):,} rows")

# Handle duplicate columns
duplicate_cols = [col for col in df_merged.columns if col.endswith('_weather') or col.endswith('_pop') or 
                  col.endswith('_ndvi') or col.endswith('_hol')]
if duplicate_cols:
    print(f"\n  Dropping {len(duplicate_cols)} duplicate columns from merging")
    df_merged = df_merged.drop(columns=duplicate_cols)

print(f"\n✓ Final merged dataset: {len(df_merged):,} rows, {len(df_merged.columns)} columns")

#%% Prepare features and encode categorical variables
print("\n" + "="*80)
print("PREPARING FEATURES")
print("="*80)

# Drop ID column
if 'ID' in df_merged.columns:
    df_merged = df_merged.drop('ID', axis=1)
    print("\n✓ Dropped ID column")

# Encode stasiun
stasiun_mapping = {station: idx for idx, station in enumerate(sorted(df_merged['stasiun'].unique()))}
df_merged['stasiun_encoded'] = df_merged['stasiun'].map(stasiun_mapping)
print(f"✓ Encoded 'stasiun': {stasiun_mapping}")

# Extract temporal features if not present
if 'month' not in df_merged.columns:
    df_merged['month'] = df_merged['tanggal'].dt.month
    df_merged['quarter'] = df_merged['tanggal'].dt.quarter
    df_merged['day_of_year'] = df_merged['tanggal'].dt.dayofyear
    df_merged['month_sin'] = np.sin(2 * np.pi * df_merged['month'] / 12)
    df_merged['month_cos'] = np.cos(2 * np.pi * df_merged['month'] / 12)
    print("✓ Extracted temporal features")

# Prepare target variable (kategori)
# Map to simplified categories: SANGAT TIDAK SEHAT and BERBAHAYA → TIDAK SEHAT
df_merged['kategori_simplified'] = df_merged['kategori'].apply(
    lambda x: 'TIDAK SEHAT' if x in ['SANGAT TIDAK SEHAT', 'BERBAHAYA', 'TIDAK SEHAT'] else x
)

# Encode kategori for classification
kategori_mapping = {'BAIK': 0, 'SEDANG': 1, 'TIDAK SEHAT': 2}
kategori_mapping_inv = {v: k for k, v in kategori_mapping.items()}
df_merged['kategori_encoded'] = df_merged['kategori_simplified'].map(kategori_mapping)
print(f"✓ Encoded 'kategori': {kategori_mapping}")

# Identify feature columns  
meta_cols = ['tanggal', 'stasiun', 'periode_data', 'kategori', 'kategori_simplified', 'kategori_encoded']
exclude_patterns = ['nama_libur', 'day_name', 'wind_direction_cardinal']
pollutant_cols = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 
                  'karbon_monoksida', 'ozon', 'nitrogen_dioksida']

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

# Get all columns
all_cols = df_merged.columns.tolist()
feature_cols = [col for col in all_cols if col not in meta_cols and 
                col not in pollutant_cols and  # Exclude direct pollutants (use their features instead)
                col not in low_corr_features and  # Exclude low-correlation features
                not any(pattern in col for pattern in exclude_patterns)]

# Keep only numeric features
numeric_features = []
for col in feature_cols:
    if df_merged[col].dtype in ['int64', 'float64', 'int8', 'float32', 'Int64', 'int16', 'float16']:
        numeric_features.append(col)

print(f"\n✓ Total features: {len(numeric_features)}")
print(f"  (Excluding direct pollutant values, using their engineered features)")

#%% Split data
print("\n" + "="*80)
print("SPLITTING DATA")
print("="*80)

# Training data
train_data = df_merged[df_merged['tanggal'] <= train_end_date].copy()
# Remove rows with missing kategori
train_data = train_data.dropna(subset=['kategori_encoded'])

print(f"\nTraining data:")
print(f"  Date range: {train_data['tanggal'].min()} to {train_data['tanggal'].max()}")
print(f"  Total rows: {len(train_data):,}")
print(f"\nCategory distribution in training data:")
print(train_data['kategori_simplified'].value_counts())
print(f"\nCategory distribution in training data:")
print(train_data['kategori_simplified'].value_counts())

# Prepare training features and target
X_train = train_data[numeric_features].copy()
y_train = train_data['kategori_encoded'].copy()

# Handle missing values in features
print(f"\nHandling missing values in features...")
# Fill NaN with median
for col in X_train.columns:
    if X_train[col].isna().sum() > 0:
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0
        X_train[col] = X_train[col].fillna(median_val)

# Replace inf with large finite values
X_train = X_train.replace([np.inf], 1e10)
X_train = X_train.replace([-np.inf], -1e10)

print(f"✓ Training set prepared: {X_train.shape}")

#%% Train classification model
print("\n" + "="*80)
print("TRAINING CLASSIFICATION MODEL")
print("="*80)

# Split training data for validation
from sklearn.model_selection import train_test_split

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"\nTraining set: {X_train_split.shape[0]:,} samples")
print(f"Validation set: {X_val_split.shape[0]:,} samples")

# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train_split)

print(f"\nClass distribution in training:")
for kategori, encoded in kategori_mapping.items():
    count = (y_train_split == encoded).sum()
    pct = count / len(y_train_split) * 100
    print(f"  {kategori}: {count:,} ({pct:.1f}%)")

# Train XGBoost Classifier
print("\nTraining XGBoost Classifier with balanced class weights...")
clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    tree_method='hist'
)

clf.fit(
    X_train_split, y_train_split,
    sample_weight=sample_weights,
    eval_set=[(X_val_split, y_val_split)],
    verbose=False
)

print("✓ Model trained successfully")

# Evaluate on validation set
y_val_pred = clf.predict(X_val_split)
val_accuracy = accuracy_score(y_val_split, y_val_pred)

print(f"\n{'='*60}")
print(f"VALIDATION RESULTS")
print(f"{'='*60}")
print(f"\nAccuracy: {val_accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_val_split, y_val_pred, 
                           target_names=list(kategori_mapping.keys())))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_val_split, y_val_pred)
cm_df = pd.DataFrame(cm, 
                     index=list(kategori_mapping.keys()),
                     columns=list(kategori_mapping.keys()))
print(cm_df)

# Feature importance
print(f"\nTop 20 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': numeric_features,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

#%% Prepare forecast period data using predicted pollutants
print("\n" + "="*80)
print("PREPARING FORECAST DATA WITH PREDICTED POLLUTANTS")
print("="*80)

# Load predicted pollutant values
print(f"\nLoading predicted pollutants from: {pollutant_predictions_file}")
df_predictions = pd.read_csv(pollutant_predictions_file)
df_predictions['tanggal'] = pd.to_datetime(df_predictions['tanggal'])

print(f"✓ Loaded {len(df_predictions):,} predictions")
print(f"  Columns: {list(df_predictions.columns)}")

# Pollutant columns
pollutant_cols = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 
                  'karbon_monoksida', 'ozon', 'nitrogen_dioksida']

# Start with predicted pollutants
df_forecast = df_predictions[['tanggal', 'stasiun'] + pollutant_cols].copy()

print(f"\n✓ Forecast skeleton: {len(df_forecast):,} rows")
print(f"  Date range: {df_forecast['tanggal'].min()} to {df_forecast['tanggal'].max()}")
print(f"  Stations: {sorted(df_forecast['stasiun'].unique())}")

# Encode stasiun
df_forecast['stasiun_encoded'] = df_forecast['stasiun'].map(stasiun_mapping)

# Extract temporal features
df_forecast['month'] = df_forecast['tanggal'].dt.month
df_forecast['quarter'] = df_forecast['tanggal'].dt.quarter
df_forecast['day_of_year'] = df_forecast['tanggal'].dt.dayofyear
df_forecast['month_sin'] = np.sin(2 * np.pi * df_forecast['month'] / 12)
df_forecast['month_cos'] = np.cos(2 * np.pi * df_forecast['month'] / 12)

# Engineer features from predicted pollutants
print("\nEngineering features from predicted pollutants...")

# Combine with historical data for rolling calculations
historical_data = train_data[['tanggal', 'stasiun'] + pollutant_cols].copy()
combined_data = pd.concat([historical_data, df_forecast[['tanggal', 'stasiun'] + pollutant_cols]], ignore_index=True)
combined_data = combined_data.sort_values(['stasiun', 'tanggal']).reset_index(drop=True)

grouped = combined_data.groupby('stasiun')

# Create key engineered features from predicted pollutants
print("  Creating rolling means (7d, 14d, 30d)...")
for pollutant in pollutant_cols:
    combined_grouped = grouped[pollutant]
    rollmean_7d = combined_grouped.transform(lambda x: x.rolling(7, min_periods=1).mean())
    rollmean_14d = combined_grouped.transform(lambda x: x.rolling(14, min_periods=1).mean())
    rollmean_30d = combined_grouped.transform(lambda x: x.rolling(30, min_periods=1).mean())
    
    df_forecast[f'{pollutant}_rollmean_7d'] = rollmean_7d.iloc[-len(df_forecast):].values
    df_forecast[f'{pollutant}_rollmean_14d'] = rollmean_14d.iloc[-len(df_forecast):].values
    df_forecast[f'{pollutant}_rollmean_30d'] = rollmean_30d.iloc[-len(df_forecast):].values

print("  Creating spikes (90th percentile)...")
for pollutant in pollutant_cols:
    threshold = train_data[pollutant].quantile(0.9)
    df_forecast[f'{pollutant}_spike'] = (df_forecast[pollutant] > threshold).astype('int8')

print("  Creating composite features...")
# PM total
df_forecast['pm_total'] = df_forecast['pm_sepuluh'] + df_forecast['pm_duakomalima']

# AQI proxy (normalized sum)
df_forecast['aqi_proxy'] = (
    df_forecast['pm_duakomalima'] / 55.5 +  # SEDANG threshold
    df_forecast['pm_sepuluh'] / 150 +
    df_forecast['ozon'] / 235
) / 3

# Gaseous pollutant index
df_forecast['gaseous_pollutant_index'] = (
    df_forecast['sulfur_dioksida'] + 
    df_forecast['nitrogen_dioksida']
) / 2

# Elevated pollutant count
spike_cols = [f'{col}_spike' for col in pollutant_cols]
df_forecast['elevated_pollutant_count'] = df_forecast[spike_cols].sum(axis=1)

# PM fine ratio
df_forecast['pm_fine_ratio'] = df_forecast['pm_duakomalima'] / (df_forecast['pm_sepuluh'] + 1e-5)

# Unhealthy flag - use rule-based approach for forecast
# Define as any pollutant exceeding "TIDAK SEHAT" threshold
df_forecast['is_unhealthy'] = (
    (df_forecast['pm_sepuluh'] > 150) |
    (df_forecast['pm_duakomalima'] > 55.4) |
    (df_forecast['sulfur_dioksida'] > 180) |
    (df_forecast['karbon_monoksida'] > 8000) |
    (df_forecast['ozon'] > 235) |
    (df_forecast['nitrogen_dioksida'] > 200)
).astype('int8')

# Rolling unhealthy days
combined_data_with_health = combined_data.copy()
combined_data_with_health['is_unhealthy'] = (
    (combined_data_with_health['pm_sepuluh'] > 150) |
    (combined_data_with_health['pm_duakomalima'] > 55.4) |
    (combined_data_with_health['sulfur_dioksida'] > 180) |
    (combined_data_with_health['karbon_monoksida'] > 8000) |
    (combined_data_with_health['ozon'] > 235) |
    (combined_data_with_health['nitrogen_dioksida'] > 200)
).astype('int8')

grouped_health = combined_data_with_health.groupby('stasiun')['is_unhealthy']
df_forecast['unhealthy_days_7d'] = grouped_health.transform(
    lambda x: x.rolling(7, min_periods=1).sum()
).iloc[-len(df_forecast):].values
df_forecast['unhealthy_days_30d'] = grouped_health.transform(
    lambda x: x.rolling(30, min_periods=1).sum()
).iloc[-len(df_forecast):].values

print("✓ Feature engineering complete")

# Get remaining features from last available data
print("\nMerging static features from last available data...")
temporal_features = ['month', 'quarter', 'day_of_year', 'month_sin', 'month_cos', 'stasiun_encoded']
engineered_pollutant_features = [col for col in df_forecast.columns if any(p in col for p in pollutant_cols)]
existing_features = temporal_features + engineered_pollutant_features + ['pm_total', 'aqi_proxy', 
                                                                          'gaseous_pollutant_index', 'is_unhealthy',
                                                                          'unhealthy_days_7d', 'unhealthy_days_30d',
                                                                          'elevated_pollutant_count', 'pm_fine_ratio']

features_to_merge = [f for f in numeric_features if f not in existing_features]

# Get last available static features (weather, population, ndvi, etc.)
stations = sorted(df_forecast['stasiun'].unique())
for station in stations:
    last_available = train_data[train_data['stasiun'] == station].sort_values('tanggal').iloc[-1]
    
    for feature in features_to_merge:
        if feature in last_available.index:
            df_forecast.loc[df_forecast['stasiun'] == station, feature] = last_available[feature]

print(f"✓ Forecast data prepared: {df_forecast.shape}")
print(f"\nSample feature values for first prediction:")
print(f"  PM2.5: {df_forecast['pm_duakomalima'].iloc[0]:.2f}")
print(f"  PM10: {df_forecast['pm_sepuluh'].iloc[0]:.2f}")
print(f"  Ozon: {df_forecast['ozon'].iloc[0]:.2f}")
print(f"  AQI Proxy: {df_forecast['aqi_proxy'].iloc[0]:.3f}")
print(f"  is_unhealthy: {df_forecast['is_unhealthy'].iloc[0]}")

#/% Make predictions
print("\n" + "="*80)
print("MAKING PREDICTIONS")
print("="*80)

# Prepare features for prediction
X_forecast = df_forecast[numeric_features].copy()

# Handle missing values
for col in X_forecast.columns:
    if X_forecast[col].isna().sum() > 0:
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0
        X_forecast[col] = X_forecast[col].fillna(median_val)

# Replace inf
X_forecast = X_forecast.replace([np.inf], 1e10)
X_forecast = X_forecast.replace([-np.inf], -1e10)

print(f"\nPredicting categories for {len(X_forecast):,} rows...")
y_forecast_encoded = clf.predict(X_forecast)
y_forecast_proba = clf.predict_proba(X_forecast)

# Decode predictions
df_forecast['kategori_encoded'] = y_forecast_encoded
df_forecast['kategori'] = df_forecast['kategori_encoded'].map(kategori_mapping_inv)

# Add confidence scores
for idx, kategori_name in kategori_mapping_inv.items():
    df_forecast[f'prob_{kategori_name.lower().replace(" ", "_")}'] = y_forecast_proba[:, idx]

print("✓ Predictions completed")

print(f"\nPredicted category distribution:")
print(df_forecast['kategori'].value_counts())

#%% Create output
print("\n" + "="*80)
print("CREATING OUTPUT")
print("="*80)

# Create ID column
df_forecast['ID'] = df_forecast['tanggal'].dt.strftime('%Y-%m-%d') + '_' + df_forecast['stasiun']

# Select output columns
output_cols = ['ID', 'tanggal', 'stasiun', 'kategori', 
               'prob_baik', 'prob_sedang', 'prob_tidak_sehat']

df_output = df_forecast[output_cols].copy()

print(f"\nOutput shape: {df_output.shape}")
print(f"Columns: {list(df_output.columns)}")

# Show distribution by station
print(f"\nCategory distribution by station:")
for station in sorted(df_output['stasiun'].unique()):
    station_data = df_output[df_output['stasiun'] == station]
    print(f"\n{station}:")
    kategori_counts = station_data['kategori'].value_counts()
    total = len(station_data)
    for kategori, count in kategori_counts.items():
        pct = (count / total) * 100
        print(f"  {kategori:20s}: {count:3d} days ({pct:5.1f}%)")

#%% Save outputs
print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

# Save detailed predictions
df_output.to_csv(output_with_kategori, index=False)
print(f"\n✓ Detailed predictions saved to: {output_with_kategori}")
print(f"  Total rows: {len(df_output):,}")

# Create submission file
df_submission = df_output[['ID', 'kategori']].copy()
df_submission = df_submission.rename(columns={'ID': 'id', 'kategori': 'category'})
df_submission.to_csv(submission_file, index=False)
print(f"\n✓ Submission file saved to: {submission_file}")
print(f"  Format: id, category")
print(f"  Total rows: {len(df_submission):,}")

# Save model statistics
model_stats = {
    'metric': ['validation_accuracy', 'training_samples', 'validation_samples', 
               'num_features', 'num_classes'],
    'value': [val_accuracy, len(X_train_split), len(X_val_split), 
              len(numeric_features), len(kategori_mapping)]
}
pd.DataFrame(model_stats).to_csv(model_stats_file, index=False)
print(f"\n✓ Model statistics saved to: {model_stats_file}")

# Display sample predictions
print(f"\n{'='*80}")
print("SAMPLE PREDICTIONS")
print(f"{'='*80}")

print("\nFirst 10 predictions:")
print(df_submission.head(10).to_string(index=False))

print("\nLast 10 predictions:")
print(df_submission.tail(10).to_string(index=False))

print("\n" + "="*80)
print("CLASSIFICATION MODEL COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nModel: XGBoost Classifier")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Features used: {len(numeric_features)}")
print(f"Total predictions: {len(df_output):,}")
print(f"Date range: {df_output['tanggal'].min()} to {df_output['tanggal'].max()}")
print(f"Categories: BAIK, SEDANG, TIDAK SEHAT")
