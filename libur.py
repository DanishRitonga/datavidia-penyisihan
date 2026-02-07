#%%
import pandas as pd
from pathlib import Path
from functools import reduce

#%%
dir = Path('data/libur-nasional/dataset-libur-nasional-dan-weekend.csv')

df = pd.read_csv(dir)