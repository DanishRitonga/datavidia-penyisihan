#%% imports
from numpy import int16
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
df = df.dropna(subset=['tanggal', 'stasiun', 'ndvi'])

# Sort for safety
df = df.sort_values(['stasiun', 'tanggal'])

#%%
daily_index = pd.date_range(start='2010-01-01', end='2025-08-31', freq='D')
df_pivoted = df.pivot(index='tanggal', columns='stasiun', values='ndvi')


ndvi_daily_wide = df_pivoted.reindex(daily_index, method='ffill')
# Optional:
ndvi_daily_wide = ndvi_daily_wide.bfill()

#%%
ndvi_long = ndvi_daily_wide.reset_index().melt(
    id_vars=['index'],
    value_vars=ndvi_daily_wide.columns,
    var_name='stasiun',
    value_name='ndvi'
)

ndvi_long = ndvi_long.rename(columns={'index': 'tanggal'})

for station in ndvi_daily_wide.columns:
    # Get actual observation dates for this station
    obs_dates = df_pivoted[station].dropna().index
    
    # Create mask for this station
    mask = ndvi_long['stasiun'] == station
    
    # Compute days since last update
    ndvi_long.loc[mask, 'days_since_ndvi_update'] = \
        ndvi_long.loc[mask, 'tanggal'].apply(
            lambda d: (d - max(dt for dt in obs_dates if dt <= d)).days
        )
ndvi_long['days_since_ndvi_update'] = ndvi_long['days_since_ndvi_update'].astype('int8')

#%%
ndvi_long.to_csv(Path('data/cleaned/ndvi_2010_2025-v2.csv'), index=False)