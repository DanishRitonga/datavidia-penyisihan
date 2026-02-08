import pandas as pd
import numpy as np

# Load historical data
df = pd.read_csv('data/cleaned/ISPU_with_features.csv')
df['tanggal'] = pd.to_datetime(df['tanggal'])

# Focus on recent data (2024-2025)
recent = df[df['tanggal'] >= '2024-01-01'].copy()

# Map kategori to numeric score
kategori_map = {
    'BAIK': 0,
    'SEDANG': 1, 
    'TIDAK SEHAT': 2,
    'SANGAT TIDAK SEHAT': 3,
    'BERBAHAYA': 4
}
recent['kategori_score'] = recent['kategori'].map(kategori_map)

print('='*80)
print('HISTORICAL AIR QUALITY BY STATION (2024-2025)')
print('='*80)

station_summary = []

for station in sorted(recent['stasiun'].unique()):
    station_data = recent[recent['stasiun'] == station]
    
    print(f'\n{station}:')
    print(f"  Total records: {len(station_data)}")
    print(f"  PM2.5 mean: {station_data['pm_duakomalima'].mean():.2f} µg/m³")
    print(f"  PM10 mean: {station_data['pm_sepuluh'].mean():.2f} µg/m³")
    print(f"  Ozon mean: {station_data['ozon'].mean():.2f} µg/m³")
    
    print(f"\n  Kategori distribution:")
    for kat in ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT', 'BERBAHAYA']:
        count = (station_data['kategori'] == kat).sum()
        pct = count / len(station_data) * 100 if len(station_data) > 0 else 0
        if count > 0:
            print(f"    {kat:25s}: {count:4d} ({pct:5.1f}%)")
    
    avg_score = station_data['kategori_score'].mean()
    print(f"\n  Average kategori score: {avg_score:.3f} (0=BAIK, 4=BERBAHAYA, lower=better)")
    
    station_summary.append({
        'stasiun': station,
        'pm25_mean': station_data['pm_duakomalima'].mean(),
        'pm10_mean': station_data['pm_sepuluh'].mean(),
        'avg_kategori_score': avg_score,
        'pct_tidak_sehat_plus': ((station_data['kategori_score'] >= 2).sum() / len(station_data) * 100)
    })

print('\n' + '='*80)
print('STATION RANKING (BEST TO WORST AIR QUALITY)')
print('='*80)

summary_df = pd.DataFrame(station_summary).sort_values('avg_kategori_score')

print("\nBy average kategori score:")
for idx, row in summary_df.iterrows():
    print(f"  {idx+1}. {row['stasiun']}: score={row['avg_kategori_score']:.3f}, "
          f"PM2.5={row['pm25_mean']:.1f}, "
          f"unhealthy+={row['pct_tidak_sehat_plus']:.1f}%")

# Also check our predictions
print('\n' + '='*80)
print('CURRENT PREDICTIONS (2025-09 to 2025-11)')
print('='*80)

pred = pd.read_csv('forecasting_predictions_2025-09-to-11.csv')
pred['tanggal'] = pd.to_datetime(pred['tanggal'])

print("\nPredicted pollutant levels by station:")
for station in sorted(pred['stasiun'].unique()):
    station_pred = pred[pred['stasiun'] == station]
    print(f'\n{station}:')
    print(f"  PM2.5 mean: {station_pred['pm_duakomalima'].mean():.2f} µg/m³")
    print(f"  PM10 mean: {station_pred['pm_sepuluh'].mean():.2f} µg/m³")
    print(f"  Ozon mean: {station_pred['ozon'].mean():.2f} µg/m³")

print('\n' + '='*80)
