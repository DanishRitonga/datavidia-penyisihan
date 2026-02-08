#%% imports
import pandas as pd
from pathlib import Path

#%% 
# ── ALL CSV LOADING BLOCKS REMAIN UNCHANGED ──

dir = Path('data/jumlah-penduduk/data-jumlah-penduduk-provinsi-dki-jakarta-berdasarkan-kelompok-usia-dan-jenis-kelamin-tahun-2013-2021-komponen-data.csv')
df = pd.read_csv(dir)

stasiun_mapping = {
    'DKI1': 'Jakarta Pusat',
    'DKI2': 'Jakarta Utara',
    'DKI3': 'Jakarta Selatan',
    'DKI4': 'Jakarta Timur',
    'DKI5': 'Jakarta Barat'
}

#%%
df['stasiun'] = df['nama_kabupaten_kota'].str.title().map({v: k for k, v in stasiun_mapping.items()})

#%%
df_agg = df.groupby(['tahun', 'stasiun'])['jumlah_penduduk'].sum().reset_index()

df_clean = df_agg[['stasiun', 'tahun', 'jumlah_penduduk']].sort_values(by=['tahun', 'stasiun']).reset_index(drop=True)
df_clean['tanggal'] = pd.to_datetime(df_clean['tahun'].astype(str) + '-12-15', format='%Y-%m-%d', errors='coerce')
df_2013_2016 = df_clean[['stasiun', 'tanggal', 'jumlah_penduduk']].rename(columns={'jumlah_penduduk': 'jumlah'})

#%% load 2017-2022 population data
years = [i for i in range(2017, 2023)]
dfs = []

for year in years:
    df1 = pd.read_csv(Path(f'data/population/Jumlah Penduduk Menurut Kabupaten_Kota di Provinsi DKI Jakarta , {year}.csv')).iloc[3:].reset_index(drop=True)
    df1.columns = ['kota', 'jumlah']
    df1['tahun'] = year
    df1['stasiun'] = df1['kota'].str.title().map({v: k for k, v in stasiun_mapping.items()})
    dfs.append(df1[['stasiun', 'tahun', 'jumlah']])

dfs1 = pd.concat(dfs).reset_index(drop=True).dropna()
dfs1['tanggal'] = pd.to_datetime(
    pd.DataFrame({
        'year': dfs1['tahun'],
        'month': 6,
        'day': 15
    })
)

df_2017_2022 = dfs1[['stasiun', 'tanggal', 'jumlah']]

#%%
df2 = pd.read_csv(Path('data/population/penduduk_2024-2025.csv'))
df2['stasiun'] = df2['NAMA KAB'].str.title().map({v: k for k, v in stasiun_mapping.items()})

df2['tanggal'] = pd.to_datetime(
    pd.DataFrame({
        'year': df2['TAHUN'],
        'month': df2['SEMESTER'].map({1: 3, 2: 9}),
        'day': 15
    })
)

df_2024_2025 = df2.groupby(['stasiun', 'tanggal'])['JML'].sum().reset_index()
df_2024_2025 = df_2024_2025.rename(columns={'JML': 'jumlah'})

#%%
df3 = pd.read_csv(Path('data/population/penduduk_2010_2013.csv'))
df3['stasiun'] = df3['KAB'].str.title().map({v: k for k, v in stasiun_mapping.items()})

df3['tanggal'] = pd.to_datetime(
    pd.DataFrame({
        'year': df3['TAHUN'],
        'month': 1,
        'day': 1
    })
)

df_2010_2013 = df3[['tanggal', 'stasiun', 'JML']].rename(columns={'JML': 'jumlah'})

#%% ── Combine all sources ──
df_all_years = pd.concat([
    df_2010_2013,
    df_2013_2016,
    df_2017_2022,
    df_2024_2025
], ignore_index=True)

df_all_years['populasi'] = pd.to_numeric(df_all_years['jumlah'], errors='coerce')
df_all_years = df_all_years.dropna(subset=['stasiun', 'tanggal', 'populasi'])
df_all_years = df_all_years[['tanggal', 'stasiun', 'populasi']].sort_values(['stasiun', 'tanggal'])

#%% Create full daily skeleton per station
stations = sorted(df_all_years['stasiun'].unique())
daily_index = pd.date_range(start='2010-01-01', end='2025-08-31', freq='D')

pop_daily_wide = pd.DataFrame(index=daily_index)

for st in stations:
    sub = df_all_years[df_all_years['stasiun'] == st].set_index('tanggal')['populasi']
    # Reindex + forward fill (main strategy for population)
    pop_daily_wide[st] = sub.reindex(daily_index).ffill()
    # Backfill very beginning if needed (small effect)
    pop_daily_wide[st] = pop_daily_wide[st].bfill()

#%% Add staleness feature (very valuable for modeling)
for st in stations:
    # Real observation dates for this station
    obs_dates = df_all_years[df_all_years['stasiun'] == st]['tanggal']
    pop_daily_wide[f'{st}_days_since_update'] = pop_daily_wide.index.to_series().apply(
        lambda d: (d - obs_dates[obs_dates <= d].max()).days if any(obs_dates <= d) else -1
    )

#%% Convert to long format (ready for merging with air quality data)
pop_long = pop_daily_wide.reset_index().melt(
    id_vars=['index'],
    value_vars=stations,
    var_name='stasiun',
    value_name='populasi'
).rename(columns={'index': 'tanggal'})

# Merge staleness columns
staleness_cols = [f'{st}_days_since_update' for st in stations]
staleness_long = pop_daily_wide[staleness_cols].reset_index().melt(
    id_vars=['index'],
    value_vars=staleness_cols,
    var_name='stasiun_temp',
    value_name='days_since_pop_update'
)
staleness_long['stasiun'] = staleness_long['stasiun_temp'].str.replace('_days_since_update', '')
staleness_long = staleness_long.drop(columns=['stasiun_temp']).rename(columns={'index': 'tanggal'})

pop_long = pop_long.merge(staleness_long, on=['tanggal', 'stasiun'], how='left')

# Final cleaning
pop_long = pop_long.sort_values(['tanggal', 'stasiun']).reset_index(drop=True)
pop_long['populasi'] = pop_long['populasi'].astype('Int64')          # nullable int
pop_long['days_since_pop_update'] = pop_long['days_since_pop_update'].astype('Int64')


#%% Save
pop_long.to_csv(Path('populasi_2010_2025_daily_long.csv'), index=False)
# df_all_years.to_csv(Path('populasi_terdata_2010_2025.csv'), index=False)