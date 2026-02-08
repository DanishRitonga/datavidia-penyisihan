import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("CORRELATION ANALYSIS FOR FEATURE SELECTION")
print("="*80)

# Load the merged training data (same as forecasting.py)
ispu_file = Path('data/cleaned/ISPU_with_features.csv')
weather_file = Path('data/cleaned/weather_with_features.csv')
population_file = Path('data/cleaned/population_with_features.csv')
ndvi_file = Path('data/cleaned/ndvi_with_features.csv')
holiday_file = Path('data/libur-nasional/dataset-libur-nasional-dan-weekend.csv')

print("\nLoading datasets...")
df_ispu = pd.read_csv(ispu_file)
df_ispu['tanggal'] = pd.to_datetime(df_ispu['tanggal'])

df_weather = pd.read_csv(weather_file)
df_weather['tanggal'] = pd.to_datetime(df_weather['tanggal'])

df_population = pd.read_csv(population_file)
df_population['tanggal'] = pd.to_datetime(df_population['tanggal'])

df_ndvi = pd.read_csv(ndvi_file)
df_ndvi['tanggal'] = pd.to_datetime(df_ndvi['tanggal'])

df_holiday = pd.read_csv(holiday_file)
df_holiday['tanggal'] = pd.to_datetime(df_holiday['tanggal'])

# Merge datasets
df_weather_merge = df_weather.drop(['ID'], axis=1, errors='ignore')
df_population_merge = df_population.drop(['ID'], axis=1, errors='ignore')
df_ndvi_merge = df_ndvi.drop(['ID'], axis=1, errors='ignore')
df_holiday_merge = df_holiday[['tanggal', 'is_holiday_nasional', 'is_weekend']].copy()

df_merged = df_ispu.copy()
df_merged = df_merged.merge(df_weather_merge, on=['tanggal', 'stasiun'], how='left', suffixes=('', '_weather'))
df_merged = df_merged.merge(df_population_merge, on=['tanggal', 'stasiun'], how='left', suffixes=('', '_pop'))
df_merged = df_merged.merge(df_ndvi_merge, on=['tanggal', 'stasiun'], how='left', suffixes=('', '_ndvi'))
df_merged = df_merged.merge(df_holiday_merge, on='tanggal', how='left', suffixes=('', '_hol'))

# Drop duplicate columns
duplicate_cols = [col for col in df_merged.columns if col.endswith('_weather') or col.endswith('_pop') or 
                  col.endswith('_ndvi') or col.endswith('_hol')]
if duplicate_cols:
    df_merged = df_merged.drop(columns=duplicate_cols)

# Drop ID
if 'ID' in df_merged.columns:
    df_merged = df_merged.drop('ID', axis=1)

# Encode stasiun
stasiun_mapping = {station: idx for idx, station in enumerate(sorted(df_merged['stasiun'].unique()))}
df_merged['stasiun_encoded'] = df_merged['stasiun'].map(stasiun_mapping)

# Extract temporal features if not present
if 'day_of_week' not in df_merged.columns:
    df_merged['day_of_week'] = df_merged['tanggal'].dt.dayofweek
    df_merged['day_of_month'] = df_merged['tanggal'].dt.day
    df_merged['month'] = df_merged['tanggal'].dt.month
    df_merged['quarter'] = df_merged['tanggal'].dt.quarter
    df_merged['day_of_year'] = df_merged['tanggal'].dt.dayofyear
    df_merged['is_weekend'] = (df_merged['day_of_week'] >= 5).astype('int8')
    df_merged['day_of_week_sin'] = np.sin(2 * np.pi * df_merged['day_of_week'] / 7)
    df_merged['day_of_week_cos'] = np.cos(2 * np.pi * df_merged['day_of_week'] / 7)
    df_merged['month_sin'] = np.sin(2 * np.pi * df_merged['month'] / 12)
    df_merged['month_cos'] = np.cos(2 * np.pi * df_merged['month'] / 12)

# Prepare kategori for classification
df_merged['kategori_simplified'] = df_merged['kategori'].apply(
    lambda x: 'TIDAK SEHAT' if x in ['SANGAT TIDAK SEHAT', 'BERBAHAYA', 'TIDAK SEHAT'] else x
)
kategori_mapping = {'BAIK': 0, 'SEDANG': 1, 'TIDAK SEHAT': 2}
df_merged['kategori_encoded'] = df_merged['kategori_simplified'].map(kategori_mapping)

print(f"✓ Merged data: {df_merged.shape}")

# Define target pollutants and meta columns
target_pollutants = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 
                     'karbon_monoksida', 'ozon', 'nitrogen_dioksida']
meta_cols = ['tanggal', 'stasiun', 'periode_data', 'kategori', 'kategori_simplified', 'kategori_encoded']
exclude_patterns = ['nama_libur', 'day_name', 'wind_direction_cardinal']

# Get numeric feature columns (exclude targets for feature analysis)
all_cols = df_merged.columns.tolist()
feature_cols = [col for col in all_cols if col not in meta_cols and 
                col not in target_pollutants and
                not any(pattern in col for pattern in exclude_patterns)]

numeric_features = []
for col in feature_cols:
    if df_merged[col].dtype in ['int64', 'float64', 'int8', 'float32', 'Int64', 'int16', 'float16']:
        numeric_features.append(col)

print(f"✓ Total numeric features: {len(numeric_features)}")
print(f"✓ Target pollutants: {len(target_pollutants)}")

# Filter to training data up to 2025-08-31
train_data = df_merged[df_merged['tanggal'] <= '2025-08-31'].copy()
print(f"✓ Training data: {len(train_data):,} rows")

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Calculate correlations with each target pollutant
print("\n1. Correlations with target pollutants (for regression):")
print("-" * 80)

target_correlations = {}
for target in target_pollutants:
    if target in train_data.columns:
        target_data = train_data.dropna(subset=[target])
        correlations = {}
        
        for feature in numeric_features:
            if feature in target_data.columns:
                # Handle inf and NaN
                feature_data = target_data[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(feature_data) > 10:
                    corr = feature_data[feature].corr(feature_data[target])
                    if not np.isnan(corr):
                        correlations[feature] = corr
        
        target_correlations[target] = correlations
        
        # Show top and bottom correlations
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n{target}:")
        print("  Top 10 positive correlations:")
        for feat, corr in sorted_corr[:10]:
            if corr > 0:
                print(f"    {feat:50s}: {corr:7.4f}")
        
        print("  Top 10 negative correlations:")
        negative = [(f, c) for f, c in sorted_corr if c < 0][:10]
        for feat, corr in negative:
            print(f"    {feat:50s}: {corr:7.4f}")

# Calculate correlations with kategori (for classification)
print("\n2. Correlations with kategori_encoded (for classification):")
print("-" * 80)

kategori_data = train_data.dropna(subset=['kategori_encoded'])
kategori_correlations = {}

for feature in numeric_features:
    if feature in kategori_data.columns:
        feature_data = kategori_data[[feature, 'kategori_encoded']].replace([np.inf, -np.inf], np.nan).dropna()
        if len(feature_data) > 10:
            corr = feature_data[feature].corr(feature_data['kategori_encoded'])
            if not np.isnan(corr):
                kategori_correlations[feature] = corr

sorted_kat_corr = sorted(kategori_correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nTop 20 features by absolute correlation with kategori:")
for feat, corr in sorted_kat_corr[:20]:
    print(f"  {feat:50s}: {corr:7.4f}")

print("\n" + "="*80)
print("IDENTIFYING LOW-CORRELATION FEATURES")
print("="*80)

# Find features with low correlation across all targets
threshold = 0.1

# Track max absolute correlation for each feature
max_abs_correlations = {}

for feature in numeric_features:
    max_corr = 0.0
    
    # Check correlations with all target pollutants
    for target in target_pollutants:
        if target in target_correlations and feature in target_correlations[target]:
            corr = abs(target_correlations[target][feature])
            max_corr = max(max_corr, corr)
    
    # Check correlation with kategori
    if feature in kategori_correlations:
        corr = abs(kategori_correlations[feature])
        max_corr = max(max_corr, corr)
    
    max_abs_correlations[feature] = max_corr

# Identify features to remove (< threshold)
low_corr_features = [feat for feat, corr in max_abs_correlations.items() if corr < threshold]
high_corr_features = [feat for feat, corr in max_abs_correlations.items() if corr >= threshold]

print(f"\nFeatures with max |correlation| < {threshold}: {len(low_corr_features)}")
print(f"Features with max |correlation| >= {threshold}: {len(high_corr_features)}")

print(f"\n{len(low_corr_features)} Low-correlation features to remove:")
print("-" * 80)
for feat in sorted(low_corr_features):
    print(f"  {feat:50s}: max |corr| = {max_abs_correlations[feat]:.4f}")

print(f"\n{len(high_corr_features)} High-correlation features to keep:")
print("-" * 80)
# Show top 30
sorted_high_corr = sorted([(f, c) for f, c in max_abs_correlations.items() if c >= threshold], 
                          key=lambda x: x[1], reverse=True)
for feat, corr in sorted_high_corr[:30]:
    print(f"  {feat:50s}: max |corr| = {corr:.4f}")

# Save results
output_file = Path('feature_selection_results.csv')
results_df = pd.DataFrame([
    {'feature': feat, 'max_abs_correlation': corr, 'keep': corr >= threshold}
    for feat, corr in sorted(max_abs_correlations.items(), key=lambda x: x[1], reverse=True)
])
results_df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

# Save list of features to keep
keep_features_file = Path('features_to_keep.txt')
with open(keep_features_file, 'w') as f:
    for feat in sorted(high_corr_features):
        f.write(f"{feat}\n")
print(f"✓ Features to keep saved to: {keep_features_file}")

# Save list of features to remove
remove_features_file = Path('features_to_remove.txt')
with open(remove_features_file, 'w') as f:
    for feat in sorted(low_corr_features):
        f.write(f"{feat}\n")
print(f"✓ Features to remove saved to: {remove_features_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
