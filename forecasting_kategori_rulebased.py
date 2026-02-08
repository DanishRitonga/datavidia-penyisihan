#%% imports
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', None)

#%% Configuration
print("="*80)
print("AIR QUALITY CATEGORY PREDICTION - RULE-BASED ISPU THRESHOLDS")
print("="*80)

# File paths
input_predictions = Path('forecasting_predictions_2025-09-to-11.csv')
output_with_kategori = Path('forecasting_predictions_with_kategori_2025-09-to-11.csv')
submission_file = Path('submission.csv')

#%% ISPU Category Thresholds
kategori_thresholds = {
    'pm_sepuluh': {
        'BAIK': (0, 50),
        'SEDANG': (51, 150),
        'TIDAK SEHAT': (151, 350),
        'SANGAT TIDAK SEHAT': (351, 420),
        'BERBAHAYA': (421, float('inf'))
    },
    'pm_duakomalima': {
        'BAIK': (0, 15.5),
        'SEDANG': (15.6, 55.4),
        'TIDAK SEHAT': (55.5, 150.4),
        'SANGAT TIDAK SEHAT': (150.5, 250.4),
        'BERBAHAYA': (250.5, float('inf'))
    },
    'sulfur_dioksida': {
        'BAIK': (0, 52),
        'SEDANG': (53, 180),
        'TIDAK SEHAT': (181, 400),
        'SANGAT TIDAK SEHAT': (401, 800),
        'BERBAHAYA': (801, float('inf'))
    },
    'karbon_monoksida': {
        'BAIK': (0, 4000),
        'SEDANG': (4001, 8000),
        'TIDAK SEHAT': (8001, 15000),
        'SANGAT TIDAK SEHAT': (15001, 30000),
        'BERBAHAYA': (30001, float('inf'))
    },
    'ozon': {
        'BAIK': (0, 120),
        'SEDANG': (121, 235),
        'TIDAK SEHAT': (236, 400),
        'SANGAT TIDAK SEHAT': (401, 800),
        'BERBAHAYA': (801, float('inf'))
    },
    'nitrogen_dioksida': {
        'BAIK': (0, 80),
        'SEDANG': (81, 200),
        'TIDAK SEHAT': (201, 400),
        'SANGAT TIDAK SEHAT': (401, 1200),
        'BERBAHAYA': (1201, float('inf'))
    }
}

kategori_priority = {
    'BERBAHAYA': 5,
    'SANGAT TIDAK SEHAT': 4,
    'TIDAK SEHAT': 3,
    'SEDANG': 2,
    'BAIK': 1
}

def get_kategori_for_pollutant(value, pollutant_name):
    if pd.isna(value):
        return None
    
    if pollutant_name not in kategori_thresholds:
        return None
    
    thresholds = kategori_thresholds[pollutant_name]
    
    for kategori, (min_val, max_val) in thresholds.items():
        if min_val <= value <= max_val:
            return kategori
    
    return 'BERBAHAYA'

def get_overall_kategori(row):
    pollutants = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 
                  'karbon_monoksida', 'ozon', 'nitrogen_dioksida']
    
    categories = []
    
    for pollutant in pollutants:
        if pollutant in row and not pd.isna(row[pollutant]):
            kategori = get_kategori_for_pollutant(row[pollutant], pollutant)
            if kategori:
                categories.append(kategori)
    
    if not categories:
        return 'BAIK'
    
    worst_kategori = max(categories, key=lambda x: kategori_priority.get(x, 0))
    return worst_kategori

def map_kategori_to_simplified(kategori):
    if kategori in ['SANGAT TIDAK SEHAT', 'BERBAHAYA']:
        return 'TIDAK SEHAT'
    return kategori

#%% Load predictions
print("\nLoading predictions from:", input_predictions)
df = pd.read_csv(input_predictions)

print(f"Loaded {len(df):,} rows")
print(f"Date range: {df['tanggal'].min()} to {df['tanggal'].max()}")
print(f"Stations: {sorted(df['stasiun'].unique())}")

pollutants = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 
              'karbon_monoksida', 'ozon', 'nitrogen_dioksida']
available_pollutants = [p for p in pollutants if p in df.columns]
print(f"\nAvailable pollutants: {available_pollutants}")

#%% Show pollutant statistics
print("\n" + "="*80)
print("POLLUTANT STATISTICS")
print("="*80)

for pollutant in available_pollutants:
    print(f"\n{pollutant}:")
    print(f"  Mean: {df[pollutant].mean():.2f}")
    print(f"  Range: [{df[pollutant].min():.2f}, {df[pollutant].max():.2f}]")
    
    # Show threshold crossing
    thresholds = kategori_thresholds[pollutant]
    for kategori, (min_val, max_val) in list(thresholds.items())[:3]:
        count = ((df[pollutant] >= min_val) & (df[pollutant] <= max_val)).sum()
        pct = count / len(df) * 100
        print(f"  {kategori}: {count} days ({pct:.1f}%)")

#%% Calculate categories
print("\n" + "="*80)
print("CALCULATING CATEGORIES")
print("="*80)

print("\nCalculating category for each pollutant...")
for pollutant in available_pollutants:
    df[f'{pollutant}_kategori'] = df[pollutant].apply(
        lambda x: get_kategori_for_pollutant(x, pollutant)
    )

print("\nDetermining overall kategori (worst pollutant)...")
df['kategori_raw'] = df.apply(get_overall_kategori, axis=1)

print("\nMapping to simplified categories...")
df['kategori'] = df['kategori_raw'].apply(map_kategori_to_simplified)

#%% Display results
print("\n" + "="*80)
print("CATEGORY DISTRIBUTION")
print("="*80)

print("\nOverall category distribution (before mapping):")
print(df['kategori_raw'].value_counts().sort_index())

print("\nFinal category distribution (after mapping):")
print(df['kategori'].value_counts().sort_index())

print("\nCategory distribution by station:")
for station in sorted(df['stasiun'].unique()):
    station_data = df[df['stasiun'] == station]
    print(f"\n{station}:")
    kategori_counts = station_data['kategori'].value_counts()
    total = len(station_data)
    for kategori, count in kategori_counts.items():
        pct = (count / total) * 100
        print(f"  {kategori:20s}: {count:3d} days ({pct:5.1f}%)")
        
    # Show which pollutant causes the category
    print(f"  Pollutant levels:")
    for pollutant in ['pm_duakomalima', 'pm_sepuluh', 'ozon']:
        if pollutant in station_data.columns:
            mean_val = station_data[pollutant].mean()
            print(f"    {pollutant:20s}: {mean_val:.2f}")

#%% Save output
print("\n" + "="*80)
print("SAVING OUTPUT")
print("="*80)

output_cols = ['ID', 'tanggal', 'stasiun'] + available_pollutants + ['kategori']
df_output = df[output_cols].copy()

df_output.to_csv(output_with_kategori, index=False)
print(f"\n✓ Detailed predictions saved to: {output_with_kategori}")
print(f"  Total rows: {len(df_output):,}")

df_submission = df_output[['ID', 'kategori']].copy()
df_submission = df_submission.rename(columns={'ID': 'id', 'kategori': 'category'})

df_submission.to_csv(submission_file, index=False)
print(f"\n✓ Submission file saved to: {submission_file}")
print(f"  Format: id, category")
print(f"  Total rows: {len(df_submission):,}")

print("\n  Sample submission format:")
print(df_submission.head(10).to_string(index=False))

print("\n" + "="*80)
print("CATEGORY PREDICTION COMPLETED")
print("="*80)
