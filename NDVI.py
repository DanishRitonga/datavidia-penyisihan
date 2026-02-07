#%% imports
import pandas as pd
from pathlib import Path
from functools import reduce

#%%

dir = Path('data/NDVI (vegetation index)/indeks-ndvi-jakarta.csv')
df = pd.read_csv(dir)

#%%
df['tanggal'] = pd.to_datetime(
    df['tanggal'], 
    format='mixed', 
    errors='coerce'
)

df = df.rename(columns={'stasiun_id': 'stasiun'})

#%%
time_skeleton = pd.DataFrame()
time_skeleton['tanggal'] = pd.date_range(start='2010-01-01', end='2025-08-31', freq='D')

#%%
df_pivoted = df.dropna(subset=['stasiun']).pivot(index='tanggal', columns='stasiun', values='ndvi')
df_pivoted = reduce(lambda left, right: pd.merge(left, right, on='tanggal', how='left'), [time_skeleton, df_pivoted])
df_extended = df_pivoted.interpolate(method='linear', axis=0, limit_direction='both')
df_ex = df_extended.dropna().melt(id_vars='tanggal', var_name='stasiun', value_name='ndvi')
df_ex = df_ex.dropna(subset=['ndvi']).sort_values(by=['tanggal', 'stasiun']).reset_index(drop=True)

#%%
df_ex.to_csv(Path('data/cleaned/ndvi_2010_2025.csv'), index=False)