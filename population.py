#%% imports
import pandas as pd
from pathlib import Path
from functools import reduce

#%%
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

df_clean = df_agg[['stasiun', 'tahun', 'jumlah_penduduk']].sort_values(by=['tahun', 'stasiun']).reset_index().copy()
df_clean = df_clean.drop(columns=['index'])
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
#%%
df_2010_2025 = pd.concat([df_2010_2013, df_2013_2016, df_2017_2022, df_2024_2025]).reset_index(drop=True)
df_2010_2025['jumlah'] = pd.to_numeric(df_2010_2025['jumlah'], errors='coerce')

time_skeleton = pd.DataFrame()
time_skeleton['tanggal'] = pd.date_range(start='2010-01-01', end='2025-08-31', freq='D')
#%%
df_pivoted = df_2010_2025.dropna(subset=['stasiun']).pivot(index='tanggal', columns='stasiun', values='jumlah')
df_pivoted = reduce(lambda left, right: pd.merge(left, right, on='tanggal', how='left'), [time_skeleton, df_pivoted])
df_extended = df_pivoted.interpolate(method='linear', axis=0, limit_direction='both')

#%%
df_ex = df_extended.dropna().melt(id_vars='tanggal', var_name='stasiun', value_name='jumlah')
df_ex = df_ex.dropna(subset=['jumlah']).sort_values(by=['tanggal', 'stasiun']).reset_index(drop=True)
df_ex['jumlah'] = df_ex['jumlah'].astype('int64')
#%%
df_ex.to_csv(Path('populasi_2010_2025.csv'), index=False)