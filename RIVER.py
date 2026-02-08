# %% [markdown]
# # River Water Quality Data Cleaning
# ## DKI Jakarta - Data Integration Pipeline (2015-2024)
# 
# This script processes and integrates river water quality data from multiple sources:
# - **PDF Tables**: 2015-2019 quarterly measurements
# - **PDF Table**: 2020 single measurement (October)
# - **CSV Files**: 2022-2024 monthly measurements
# 
# **Outputs**: 
# 1. Sparse data (actual measurements): `river_water_2015-2024.csv`
# 2. Daily expanded data with staleness tracking: `data/cleaned/river_quality_2015-2025_daily.csv`

# %% [markdown]
## 1. Setup & Imports

# %%
import pandas as pd
import geopandas as gpd
import camelot
import re
import requests
from pathlib import Path
from functools import reduce
import json
from shapely.geometry import Polygon, MultiPolygon
import gc

pd.set_option('display.max_columns', None)

# %% [markdown]
## 2. Configuration

# %% [markdown]
### 2.1 Station Mapping

# %%
stasiun_mapping = {
    'DKI1': 'Jakarta Pusat',
    'DKI2': 'Jakarta Utara',
    'DKI3': 'Jakarta Selatan',
    'DKI4': 'Jakarta Timur',
    'DKI5': 'Jakarta Barat'
}

# %% [markdown]
### 2.2 Data Source Configuration

# %%
# PDF: 2015-2019 data separation (year: number of periods)
data_separation = {
    2015: 3,  # 3 periods (quarterly)
    2016: 2,
    2017: 3,
    2018: 4,
    2019: 4
}

# CSV files
csv_files = [
    ('data/kualitas-air-sungai/sungai2022.csv', ';'),
    ('data/kualitas-air-sungai/sungai2023.csv', ';'),
    ('data/kualitas-air-sungai/data-kualitas-air-sungai-komponen-data.csv', ',')
]

# %% [markdown]
### 2.3 Parameter Standardization Dictionary

# %%
river_params_dict = {
    "total_dissolved_solids": {"params1": "TDS", "params2": "TDS", "params3": "ZAT PADAT TERLARUT TDS"},
    "total_suspended_solids": {"params1": "TSS", "params2": "TSS", "params3": "ZAT PADAT TERSUSPENSI TSS"},
    "ph": {"params1": "pH", "params2": "pH", "params3": "PH"},
    "biological_oxygen_demand": {"params1": "BOD", "params2": "BOD", "params3": "BOD"},
    "chemical_oxygen_demand": {"params1": "COD", "params2": "COD", "params3": "COD DICHROMAT"},
    "cadmium": {"params1": "Cd", "params2": "Cd", "params3": "KADMIUM CD"},
    "chromium_vi": {"params1": "Cr6+", "params2": "Cr6", "params3": "CROM HEXAVALEN CR6"},
    "copper": {"params1": "Cu", "params2": "Cu", "params3": "TEMBAGA CU"},
    "lead": {"params1": "Pb", "params2": "Pb", "params3": "TIMAH HITAM PB"},
    "mercury": {"params1": "Hg", "params2": "Hg", "params3": "HG"},
    "zinc": {"params1": "Zn", "params2": "Zn", "params3": "SENG ZN"},
    "oil_and_grease": {"params1": "Minyak dan Lemak", "params2": "Minyak dan Lemak", "params3": "MINYAK DAN LEMAK"},
    "mbas_detergent": {"params1": "MBAS", "params2": "MBAS", "params3": "MBAS"},
    "total_coliform": {"params1": "Bakteri Koli", "params2": "Total Coliform", "params3": "TOTAL COLIFORM"},
    "fecal_coliform": {"params1": "Bakteri Koli Tinja", "params2": "Fecal Coliform", "params3": "FECAL COLIFORM"}
}

# Get all possible parameters from the dictionary
ALL_PARAMETERS = sorted(river_params_dict.keys())

# CSV parameter name variations (different files use different naming conventions)
# Maps: normalized_name -> standardized params3 name
CSV_PARAM_ALIASES = {
    # TDS/TSS variations
    'TDS': 'ZAT PADAT TERLARUT TDS',
    'TSS': 'ZAT PADAT TERSUSPENSI TSS',
    # COD variations
    'COD': 'COD DICHROMAT',
    # Metal symbols (short form used in 2023/2024 CSVs)
    'CD': 'KADMIUM CD',
    'CR6': 'CROM HEXAVALEN CR6',
    'CR6+': 'CROM HEXAVALEN CR6',
    'CU': 'TEMBAGA CU',
    'PB': 'TIMAH HITAM PB',
    'ZN': 'SENG ZN',
    # Coliform variations
    'FECAL COLIFORM': 'FECAL COLIFORM',
    'TOTAL COLIFORM': 'TOTAL COLIFORM',
}

# %% [markdown]
## 3. Geographic Data Loading

# %% [markdown]
### 3.1 Helper Functions for Boundary Processing

# %%
def parse_boundary_sql(sql_content):
    """Parse SQL file to extract boundary data."""
    lines = sql_content.split('\n')
    data = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("('"):
            try:
                first_quote = line.find("'")
                second_quote = line.find("'", first_quote + 1)
                kode = line[first_quote+1:second_quote]
                
                third_quote = line.find("'", second_quote + 1)
                fourth_quote = line.find("'", third_quote + 1)
                nama = line[third_quote+1:fourth_quote]
                
                after_nama = line[fourth_quote+2:]
                parts = after_nama.split(',', 2)
                
                if len(parts) >= 3:
                    lat = parts[0].strip()
                    lng = parts[1].strip()
                    rest = parts[2]
                    path_start = rest.find("'")
                    path_end = rest.rfind("'")
                    if path_start != -1 and path_end != -1:
                        geom = rest[path_start+1:path_end]
                        data.append([kode, nama, lat, lng, geom])
            except:
                pass
    
    return pd.DataFrame(data, columns=['kode', 'nama', 'lat', 'lng', 'geom'])

# %%
def swap_coords(coord_list):
    """Recursively swap [lat,lng] to [lng,lat] for GeoJSON compatibility."""
    if isinstance(coord_list[0], (int, float)):
        return [coord_list[1], coord_list[0]]
    else:
        return [swap_coords(c) for c in coord_list]

# %%
def parse_geometry(geom_str):
    """Parse geometry string to Shapely Polygon/MultiPolygon."""
    try:
        coords = json.loads(geom_str)
        coords = swap_coords(coords)
        
        if isinstance(coords[0][0], (int, float)):
            return Polygon(coords)
        elif isinstance(coords[0][0][0], (int, float)):
            exterior = coords[0]
            holes = coords[1:] if len(coords) > 1 else []
            return Polygon(exterior, holes)
        else:
            polygons = []
            for poly_coords in coords:
                exterior = poly_coords[0]
                holes = poly_coords[1:] if len(poly_coords) > 1 else []
                polygons.append(Polygon(exterior, holes))
            return MultiPolygon(polygons)
    except:
        return None

# %%
def assign_to_nearest_region(gdf_points, gdf_boundaries, nama_col='kota_kabupaten'):
    """Assign unassigned points to nearest boundary."""
    unassigned_mask = gdf_points[nama_col].isna()
    unassigned_indices = gdf_points[unassigned_mask].index
    
    if len(unassigned_indices) == 0:
        return gdf_points
    
    current_counts = gdf_points[nama_col].value_counts().to_dict()
    
    for idx in unassigned_indices:
        point = gdf_points.loc[idx, 'geometry']
        distances = []
        
        for _, boundary_row in gdf_boundaries.iterrows():
            dist = point.distance(boundary_row['geometry'])
            nama = boundary_row['nama']
            count = current_counts.get(nama, 0)
            distances.append((dist, count, nama))
        
        distances.sort(key=lambda x: (x[0], x[1]))
        _, _, nearest_nama = distances[0]
        
        gdf_points.loc[idx, nama_col] = nearest_nama
        current_counts[nearest_nama] = current_counts.get(nearest_nama, 0) + 1
    
    return gdf_points

# %%
def add_stasiun_column(df, stasiun_mapping):
    """Add 'stasiun' column based on kota_kabupaten mapping."""
    reverse_mapping = {v: k for k, v in stasiun_mapping.items()}
    df['stasiun'] = df['kota_kabupaten'].map(reverse_mapping)
    return df

# %% [markdown]
### 3.2 Load DKI Jakarta Administrative Boundaries

# %%
url = "https://raw.githubusercontent.com/cahyadsn/wilayah_boundaries/refs/heads/main/db/kab/wilayah_boundaries_kab_31.sql"
response = requests.get(url)
response.raise_for_status()
sql_content = response.text

# Clean SQL
clean_sql = re.sub(r'^/\*.*?\*/\s*', '', sql_content, flags=re.DOTALL | re.MULTILINE).strip()
clean_sql = re.sub(r'--.*?$', '', clean_sql, flags=re.MULTILINE)
clean_sql = re.sub(r'ENGINE=\w+\s*|DEFAULT CHARSET=\w+\s*', '', clean_sql, flags=re.IGNORECASE)
clean_sql = clean_sql.replace('`', '')

# %%
# Parse and filter boundaries (exclude Kepulauan Seribu)
df_batas_wilayah = parse_boundary_sql(clean_sql)
df_batas_wilayah = df_batas_wilayah[df_batas_wilayah['kode'] != '31.01'].reset_index(drop=True)

print(f"✓ Loaded {len(df_batas_wilayah)} mainland DKI Jakarta boundaries")

# %%
# Convert to GeoDataFrame
df_batas_wilayah['geometry'] = df_batas_wilayah['geom'].apply(parse_geometry)
gdf_boundaries = gpd.GeoDataFrame(
    df_batas_wilayah[df_batas_wilayah['geometry'].notna()], 
    geometry='geometry', 
    crs='EPSG:4326'
)

print(f"✓ Created GeoDataFrame with {len(gdf_boundaries)} boundaries")

# %% [markdown]
## 4. Sampling Locations Processing

# %% [markdown]
### 4.1 Extract Sampling Locations from PDF

# %%
sampling_locs = camelot.read_pdf("airsungai_20250702121933.pdf", pages="60-62", flavor='lattice')
print(f"✓ Extracted {len(sampling_locs)} location tables from PDF")

# %%
def correct_num_format(df, cols):
    """Convert European number format to standard format."""
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .replace(['', 'nan', 'None'], pd.NA)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# %%
sampling_locs_header = ["No","Kode","Sungai","Sub Jaringan","Alamat","DAS","Lintang (DD)","Bujur (DD)","Lintang (DMS)","Bujur (DMS)"]

slocs = []
for sloc in sampling_locs:
    df_sloc = sloc.df.iloc[2:]
    df_sloc.columns = sampling_locs_header
    df_sloc = df_sloc[['Kode', 'Lintang (DD)', 'Bujur (DD)']].rename(columns={'Lintang (DD)': 'lat', 'Bujur (DD)': 'lng'})
    df_sloc = correct_num_format(df_sloc, ['lat', 'lng'])
    slocs.append(df_sloc)

slocs = pd.concat(slocs, ignore_index=True)
print(f"✓ Processed {len(slocs)} sampling locations")

# %% [markdown]
### 4.2 Assign Sampling Locations to Stations

# %%
# Convert to GeoDataFrame and perform spatial join
gdf_slocs = gpd.GeoDataFrame(slocs, geometry=gpd.points_from_xy(slocs.lng, slocs.lat), crs='EPSG:4326')

gdf_slocs_with_region = gpd.sjoin(gdf_slocs, gdf_boundaries[['nama', 'geometry']], how='left', predicate='within')

if 'index_right' in gdf_slocs_with_region.columns:
    gdf_slocs_with_region = gdf_slocs_with_region.drop(columns=['index_right'])

gdf_slocs_with_region = gdf_slocs_with_region.rename(columns={'nama': 'kota_kabupaten'})

# %%
# Assign to nearest boundary and add stasiun codes
gdf_slocs_with_region = assign_to_nearest_region(gdf_slocs_with_region, gdf_boundaries, nama_col='kota_kabupaten')

slocs_with_stasiun = pd.DataFrame(gdf_slocs_with_region.drop(columns=['lat', 'lng', 'geometry']))
slocs_with_stasiun['kota_kabupaten'] = slocs_with_stasiun['kota_kabupaten'].str.replace('Kota Administrasi ', '')
slocs_with_stasiun = add_stasiun_column(slocs_with_stasiun, stasiun_mapping).drop(columns=['kota_kabupaten'])

print(f"\n✓ All {len(slocs_with_stasiun)} sampling locations assigned to stations")
print(slocs_with_stasiun['stasiun'].value_counts().sort_index())

# %% [markdown]
## 5. PDF Table Loading

# %% [markdown]
### 5.1 Load PDF Tables (2015-2019)

# %%
tables = camelot.read_pdf("airsungai_20250702121933.pdf", pages="66-113", flavor='lattice')
print(f"✓ Loaded {len(tables)} tables from PDF (2015-2019)")

# %% [markdown]
### 5.2 Load PDF Table (2020)

# %%
table2020 = camelot.read_pdf("airsungai_20250702121933.pdf", pages="115", flavor='lattice')
print(f"✓ Loaded 2020 table (split across {len(table2020)} pages)")

# %% [markdown]
## 6. Helper Functions

# %% [markdown]
### 6.1 Number Format Correction

# %%
def handle_range_values(value):
    """Handle range values like '<0.002', '>10', '≤5', etc."""
    if pd.isna(value) or value == '':
        return value
    value_str = str(value)
    value_str = re.sub(r'^[<>≤≥\s]+|[<>=≤≥\s]+$', '', value_str)
    return value_str

# %%
def correct_european_number_format(df, cols):
    """Convert European number format and handle range values."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(handle_range_values)
            df[col] = (
                df[col].astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .replace(['', 'nan', 'None'], pd.NA)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# %% [markdown]
### 6.2 Data Structure Helpers

# %%
def spatial_join_with_boundaries(df, lat_col, lng_col, gdf_boundaries, stasiun_mapping):
    """Perform spatial join to assign stations to coordinates."""
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lng_col], df[lat_col]), crs='EPSG:4326')
    
    gdf_with_region = gpd.sjoin(gdf, gdf_boundaries[['nama', 'geometry']], how='left', predicate='within')
    
    if 'index_right' in gdf_with_region.columns:
        gdf_with_region = gdf_with_region.drop(columns=['index_right'])
    
    gdf_with_region = gdf_with_region.rename(columns={'nama': 'kota_kabupaten'})
    gdf_with_region = assign_to_nearest_region(gdf_with_region, gdf_boundaries, nama_col='kota_kabupaten')
    
    df_with_region = pd.DataFrame(gdf_with_region.drop(columns=['geometry']))
    df_with_region['kota_kabupaten'] = df_with_region['kota_kabupaten'].str.replace('Kota Administrasi ', '')
    df_with_region = add_stasiun_column(df_with_region, stasiun_mapping)
    
    return df_with_region

# %%
def pivot_and_merge_parameters(df, index_columns, parameter_column, value_column, param_types):
    """Pivot and merge different parameter types."""
    dfs = []
    for param_type in param_types:
        df_filtered = df[df['jenis_parameter'] == param_type]
        df_pivoted = df_filtered.pivot_table(
            index=index_columns,
            columns=parameter_column,
            values=value_column
        )
        df_pivoted = df_pivoted.reset_index()
        dfs.append(df_pivoted)
    
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=index_columns, how='inner'), dfs)
    df_merged.columns.name = None
    
    return df_merged

# %%
def ensure_all_parameters(df, all_params):
    """Ensure dataframe has all parameters, adding missing ones as pd.NA."""
    for param in all_params:
        if param not in df.columns:
            df[param] = pd.NA
    
    # Reorder columns: tanggal, stasiun, then all parameters in sorted order
    column_order = ['tanggal', 'stasiun'] + sorted(all_params)
    return df[column_order]

# %%
def create_date_skeleton(years, stations):
    """Create skeleton dataframe with all date-station combinations."""
    skeleton_data = []
    
    for year in years:
        if year != 2025: 
            date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
        else:
            date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-08-31', freq='D')
        for date in date_range:
            for stasiun_code in stations:
                skeleton_data.append({'tanggal': date, 'stasiun': stasiun_code})
    
    skeleton_df = pd.DataFrame(skeleton_data)
    return skeleton_df.sort_values(['tanggal', 'stasiun']).reset_index(drop=True)

# %%
def expand_temporal_data(df, skeleton, value_columns):
    """Expand sparse data to daily using forward/backward fill."""
    df_expanded = skeleton.merge(df[['tanggal', 'stasiun'] + value_columns], on=['tanggal', 'stasiun'], how='left')
    df_expanded[value_columns] = df_expanded.groupby('stasiun')[value_columns].transform(lambda x: x.bfill().ffill())
    return df_expanded

# %% [markdown]
## 7. Data Processing Functions

# %% [markdown]
### 7.1 PDF Data Processing (2015-2019)

# %%
def process_pdf_river_data(tables, data_separation, slocs_with_stasiun, river_params_dict, all_params):
    """Process river quality data from PDF tables (2015-2019)."""
    numeric_cols = ["TDS", "TSS", "pH", "BOD", "COD", "Cd", "Cr6+", "Cu", "Pb", "Hg", "Zn", 
                    "Minyak dan Lemak", "MBAS", "Bakteri Koli", "Bakteri Koli Tinja"]
    headers = ["no", "Kode", "Sungai", "DAS", *numeric_cols]
    param_rename_map = {v['params1']: k for k, v in river_params_dict.items()}
    
    processed_tables = 0
    tabs = []
    
    for year, num_periods in data_separation.items():
        for period in range(1, num_periods + 1):
            table_indices = [processed_tables, processed_tables + 1, processed_tables + 2]
            period_dfs = []
            
            for idx in table_indices:
                table = tables[idx].df.iloc[5:].copy()
                table.columns = headers
                table = correct_european_number_format(table.drop(columns=["no"]), numeric_cols)
                table = table.merge(slocs_with_stasiun[['Kode', 'stasiun']], on='Kode', how='left')
                table = table.rename(columns=param_rename_map)
                period_dfs.append(table)
            
            period_combined = pd.concat(period_dfs, ignore_index=True)
            # Only aggregate parameters that exist in this dataset
            available_params = [col for col in param_rename_map.values() if col in period_combined.columns]
            period_agg = period_combined.groupby('stasiun').agg({col: 'mean' for col in available_params}).reset_index()
            period_agg['tahun'] = year
            period_agg['periode'] = period
            tabs.append(period_agg)
            processed_tables += 3
    
    river_quality_all = pd.concat(tabs, ignore_index=True)
    
    # Add date column
    month_map = {1: 3, 2: 6, 3: 9, 4: 12}
    river_quality_all['bulan'] = river_quality_all['periode'].map(month_map)
    river_quality_all['tanggal'] = pd.to_datetime(
        river_quality_all['tahun'].astype(str) + '-' + river_quality_all['bulan'].astype(str) + '-15'
    )
    
    # Keep only tanggal, stasiun and parameter columns
    keep_cols = ['tanggal', 'stasiun'] + [col for col in river_quality_all.columns 
                                           if col in all_params]
    result = river_quality_all[keep_cols]
    
    # Ensure all parameters exist
    return ensure_all_parameters(result, all_params)

# %% [markdown]
### 7.2 2020 Data Processing

# %%
def process_2020_river_data(table2020, slocs_with_stasiun, river_params_dict, all_params):
    """Process river quality data from 2020 PDF table (single October measurement)."""
    table2020_params = [
        ["Suhu", "TDS", "TSS", "pH", "BOD", "COD", "Total-P", "NO3", "Cd", "Cr6", "Cu", "Pb"], 
        ["Hg", "Zn", "Flourida", "NO2", "Klorin Bebas", "H2S", "Minyak dan Lemak", "MBAS", "Fenol", "Fecal Coliform", "Total Coliform"]
    ]
    table2020_index = ["No", "Kode", "Sungai", "DAS"]
    
    # Parse both tables
    sungai2020 = []
    for t in range(len(table2020)):
        df = table2020[t].df.iloc[5:].copy()
        df.columns = table2020_index + table2020_params[t]
        sungai2020.append(df)
    
    # Merge and process
    sungai2020_df = reduce(lambda left, right: pd.merge(left, right, on=table2020_index, how='inner'), sungai2020)
    all_raw_params = [col for sublist in table2020_params for col in sublist]
    sungai2020_df = correct_european_number_format(sungai2020_df, all_raw_params)
    sungai2020_df = sungai2020_df.merge(slocs_with_stasiun[['Kode', 'stasiun']], on='Kode', how='left')
    
    # Standardize parameter names
    param_rename_map = {v['params2']: k for k, v in river_params_dict.items()}
    sungai2020_df = sungai2020_df.rename(columns=param_rename_map)
    
    # Aggregate by station - only for available parameters
    available_params = [col for col in sungai2020_df.columns if col in all_params]
    sungai2020_agg = sungai2020_df.groupby('stasiun').agg({col: 'mean' for col in available_params}).reset_index()
    sungai2020_agg['tanggal'] = pd.to_datetime('2020-10-15')
    
    # Keep only tanggal, stasiun and available parameter columns
    result = sungai2020_agg[['tanggal', 'stasiun'] + available_params]
    
    # Ensure all parameters exist
    return ensure_all_parameters(result, all_params)

# %% [markdown]
### 7.3 CSV Data Processing (2022-2024)

# %%
def process_csv_river_data(csv_file, gdf_boundaries, stasiun_mapping, river_params_dict, all_params, delimiter=';'):
    """Process river quality data from CSV file."""
    df = pd.read_csv(csv_file, sep=delimiter)
    
    # Standardize column names
    rename_map = {}
    if 'latitude' in df.columns:
        rename_map['latitude'] = 'lintang'
    elif 'lintang_selatan' in df.columns:
        rename_map['lintang_selatan'] = 'lintang'
    
    if 'longitude' in df.columns:
        rename_map['longitude'] = 'bujur'
    elif 'bujur_timur' in df.columns:
        rename_map['bujur_timur'] = 'bujur'
    
    if 'periode_data' in df.columns and 'tahun' not in df.columns:
        rename_map['periode_data'] = 'tahun'
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    if 'periode_data' in df.columns and 'tahun' in df.columns:
        df = df.drop(columns=['periode_data'])
    
    # Normalize jenis_parameter to uppercase for consistency
    df['jenis_parameter'] = df['jenis_parameter'].str.upper()
    
    # Normalize parameter names to uppercase for consistency across CSV files
    df['parameter'] = df['parameter'].str.upper()
    
    # Correct number formats
    df = correct_european_number_format(df, ['lintang', 'bujur', 'hasil_pengukuran'])
    
    required_cols = ['tahun', 'bulan_sampling', 'lintang', 'bujur', 'jenis_parameter', 'parameter', 'hasil_pengukuran']
    df = df[required_cols]
    
    # Aggregate duplicates
    df = df.groupby(['tahun', 'bulan_sampling', 'lintang', 'bujur', 'jenis_parameter', 'parameter'], as_index=False)['hasil_pengukuran'].mean()
    
    # Pivot parameter types
    param_types = ['KIMIA', 'FISIKA', 'BIOLOGI']
    index_cols = ['tahun', 'bulan_sampling', 'lintang', 'bujur']
    df_pivoted = pivot_and_merge_parameters(df, index_cols, 'parameter', 'hasil_pengukuran', param_types)
    
    # Apply CSV parameter aliases to handle naming variations across different CSV files
    # This maps short forms (e.g., 'CD', 'COD', 'TDS') to full standardized names
    df_pivoted = df_pivoted.rename(columns=CSV_PARAM_ALIASES)
    
    # Standardize parameter names to internal representation
    param_rename_map = {v['params3']: k for k, v in river_params_dict.items()}
    df_pivoted = df_pivoted.rename(columns=param_rename_map)
    
    # Keep available parameters
    available_params = [col for col in df_pivoted.columns if col in all_params]
    df_pivoted = df_pivoted[index_cols + available_params]
    
    # Create date column
    if df_pivoted['bulan_sampling'].astype(str).str.len().max() >= 6:
        df_pivoted['tanggal'] = (
            df_pivoted['bulan_sampling'].astype(str).str[:4] + '-' + 
            df_pivoted['bulan_sampling'].astype(str).str[4:] + '-15'
        )
    else:
        df_pivoted['tanggal'] = (
            df_pivoted['tahun'].astype(str) + '-' + 
            df_pivoted['bulan_sampling'].astype(str).str.zfill(2) + '-15'
        )
    
    df_pivoted['tanggal'] = pd.to_datetime(df_pivoted['tanggal'], format='%Y-%m-%d')
    
    # Spatial join to assign stations
    df_with_stasiun = spatial_join_with_boundaries(df_pivoted, 'lintang', 'bujur', gdf_boundaries, stasiun_mapping)
    
    # Drop unnecessary columns
    cols_to_drop = ['tahun', 'bulan_sampling', 'lintang', 'bujur', 'kota_kabupaten']
    df_with_stasiun = df_with_stasiun.drop(columns=[col for col in cols_to_drop if col in df_with_stasiun.columns])
    
    # Keep only tanggal, stasiun and available parameters
    result = df_with_stasiun[['tanggal', 'stasiun'] + available_params]
    
    # Ensure all parameters exist
    return ensure_all_parameters(result, all_params)

# %% [markdown]
## 8. Unified Data Integration Pipeline

# %%
print("="*70)
print("RIVER WATER QUALITY DATA INTEGRATION PIPELINE")
print("="*70)

# %% [markdown]
### Step 1: Process PDF Data (2015-2019)

# %%
print("\n[1/6] Processing PDF data (2015-2019)...")
river_pdf = process_pdf_river_data(tables, data_separation, slocs_with_stasiun, river_params_dict, ALL_PARAMETERS)
print(f"  → Shape: {river_pdf.shape}")
print(f"  → Date range: {river_pdf['tanggal'].min()} to {river_pdf['tanggal'].max()}")
print(f"  → Total parameters: {len(ALL_PARAMETERS)}")
print(f"  → Non-null parameters: {river_pdf[ALL_PARAMETERS].notna().any().sum()}")

# %% [markdown]
### Step 2: Process 2020 Data

# %%
print("\n[2/6] Processing 2020 PDF data...")
river_2020 = process_2020_river_data(table2020, slocs_with_stasiun, river_params_dict, ALL_PARAMETERS)
print(f"  → Shape: {river_2020.shape}")
print(f"  → Date: {river_2020['tanggal'].unique()[0]}")
print(f"  → Total parameters: {len(ALL_PARAMETERS)}")
print(f"  → Non-null parameters: {river_2020[ALL_PARAMETERS].notna().any().sum()}")

# %% [markdown]
### Step 3: Process CSV Files (2022-2024)

# %%
print("\n[3/6] Processing CSV files...")
data_dir = Path('data/kualitas-air-sungai')

csv_files_full = [
    (data_dir / 'sungai2022.csv', ';'),
    (data_dir / 'sungai2023.csv', ';'),
    (data_dir / 'data-kualitas-air-sungai-komponen-data.csv', ',')
]

river_csv_list = []
for csv_file, delimiter in csv_files_full:
    df_csv = process_csv_river_data(str(csv_file), gdf_boundaries, stasiun_mapping, river_params_dict, ALL_PARAMETERS, delimiter)
    non_null_params = df_csv[ALL_PARAMETERS].notna().any().sum()
    print(f"  → {csv_file.name}: {df_csv.shape} ({non_null_params} non-null parameters)")
    river_csv_list.append(df_csv)

river_csv = pd.concat(river_csv_list, ignore_index=True)
print(f"  → Combined: {river_csv.shape}")
print(f"  → Date range: {river_csv['tanggal'].min()} to {river_csv['tanggal'].max()}")

# %% [markdown]
### Step 4: Verify Data Structures

# %%
print("\n[4/6] Verifying data structures...")

# All dataframes should now have identical columns
assert list(river_pdf.columns) == list(river_2020.columns) == list(river_csv.columns), "Column mismatch!"

print(f"  ✓ All datasets have {len(river_pdf.columns)} columns")
print(f"  ✓ Column order: ['tanggal', 'stasiun'] + {len(ALL_PARAMETERS)} parameters")

# Show parameter availability across datasets
pdf_non_null = set(river_pdf[ALL_PARAMETERS].columns[river_pdf[ALL_PARAMETERS].notna().any()])
csv_non_null = set(river_csv[ALL_PARAMETERS].columns[river_csv[ALL_PARAMETERS].notna().any()])
params_2020_non_null = set(river_2020[ALL_PARAMETERS].columns[river_2020[ALL_PARAMETERS].notna().any()])

print(f"\n  Parameters with data:")
print(f"    → PDF (2015-2019): {len(pdf_non_null)}")
print(f"    → 2020: {len(params_2020_non_null)}")
print(f"    → CSV (2022-2024): {len(csv_non_null)}")
print(f"    → All datasets: {len(pdf_non_null & params_2020_non_null & csv_non_null)}")
print(f"    → Any dataset: {len(pdf_non_null | params_2020_non_null | csv_non_null)}")

# %% [markdown]
### Step 5: Combine All Data Sources

# %%
print("\n[5/6] Combining all data sources...")

river_all = pd.concat([river_pdf, river_2020, river_csv], ignore_index=True)
river_all = river_all.sort_values(['tanggal', 'stasiun']).reset_index(drop=True)

print(f"  → Combined shape: {river_all.shape}")
print(f"  → Date range: {river_all['tanggal'].min()} to {river_all['tanggal'].max()}")
print(f"  → Unique dates: {river_all['tanggal'].nunique()}")
print(f"  → Records by station:")
for stasiun in sorted(river_all['stasiun'].unique()):
    count = (river_all['stasiun'] == stasiun).sum()
    print(f"     {stasiun}: {count}")

# %% [markdown]
### Step 6: Export Data

# %%
print("\n[6/6] Exporting data...")

value_cols = ALL_PARAMETERS

# Export the sparse/original data (actual measurements only)
river_all.to_csv('river_water_2015-2024.csv', index=False)
print(f"  ✓ Saved data to: river_water_2015-2024.csv")
print(f"  → Shape: {river_all.shape}")
print(f"  → Records: {len(river_all):,}")

# %% [markdown]
### Pipeline Summary

# %%
print("\n" + "="*70)
print("PROCESSING COMPLETE")
print("="*70)
print(f"\nDataset created:")
print(f"  river_all: {river_all.shape} (actual measurements only)")
print(f"\nDate coverage: {river_all['tanggal'].min().date()} to {river_all['tanggal'].max().date()}")
print(f"Stations: {sorted(river_all['stasiun'].unique())}")
print(f"Parameters: {len(value_cols)}")
print(f"\nNote: Data contains only actual measurements (sparse/irregular dates).")
print(f"      No daily interpolation or temporal expansion applied.")
print("="*70)

# %% [markdown]
## 9. Data Quality Check

# %%
print("="*70)
print("DATA QUALITY CHECK")
print("="*70)

# %%
# Missing values in sparse data
print("\n[1] Missing values in river_all (sparse data):")
missing_counts = river_all[value_cols].isnull().sum()
missing_pct = (missing_counts / len(river_all) * 100).round(2)
missing_df = pd.DataFrame({
    'Parameter': missing_counts.index,
    'Missing Count': missing_counts.values,
    'Missing %': missing_pct.values
}).sort_values('Missing %', ascending=False)

print(missing_df.head(10).to_string(index=False))

# %%
# Data sparsity analysis
print("\n[2] Data sparsity by parameter:")
non_null_counts = river_all[value_cols].notna().sum()
total_records = len(river_all)
print(f"\n{'Parameter':<35} {'Non-null Count':<15} {'Coverage %':<12}")
print("-" * 62)
for param in sorted(value_cols):
    count = non_null_counts[param]
    pct = (count / total_records * 100) if total_records > 0 else 0
    print(f"{param:<35} {count:<15,} {pct:>10.2f}%")

# %%
# Data completeness by year
print("\n[3] Data completeness by year:")
river_all_check = river_all.copy()
river_all_check['year'] = river_all_check['tanggal'].dt.year
completeness = river_all_check.groupby('year').agg({
    'tanggal': 'count',
    'stasiun': lambda x: x.nunique()
}).rename(columns={'tanggal': 'records', 'stasiun': 'stations'})
print(completeness)

# %%
# Duplicate records
print("\n[4] Duplicate records check:")
duplicates = river_all[river_all.duplicated(subset=['tanggal', 'stasiun'], keep=False)]
print(f"  → Duplicate date-station pairs: {len(duplicates)}")
if len(duplicates) > 0:
    print("\n  Sample duplicates:")
    print(duplicates[['tanggal', 'stasiun']].head(10).to_string(index=False))

print("\n" + "="*70)

# %% [markdown]
## 10. Sophisticated Data Expansion Pipeline

# %% [markdown]
### Phase 1 - Clean & Understand Reality

# %%
import numpy as np

print("="*70)
print("PHASE 1: DATA EXPLORATION & REGIME ANALYSIS")
print("="*70)

# %% [markdown]
#### 1.1 Basic Inspection

# %%
print("\n[1.1] Basic Dataset Inspection")
print("-" * 70)
print(f"Shape: {river_all.shape}")
print(f"Date range: {river_all['tanggal'].min().date()} → {river_all['tanggal'].max().date()}")
print(f"Number of stations: {river_all['stasiun'].nunique()}")
print(f"Unique sampling dates: {river_all['tanggal'].nunique()}")

print("\n Station distribution:")
print(river_all['stasiun'].value_counts().sort_index())

# %% [markdown]
#### 1.2 Detect Regime Changes

# %%
print("\n[1.2] Regime Change Detection")
print("-" * 70)

# Analyze by year
regime_analysis = river_all.groupby(river_all['tanggal'].dt.year).agg({
    'tanggal': lambda x: x.nunique(),  # unique dates
    'stasiun': 'count'  # total rows
}).rename(columns={'tanggal': 'unique_dates', 'stasiun': 'total_rows'})

regime_analysis['rows_per_date'] = (regime_analysis['total_rows'] / regime_analysis['unique_dates']).round(1)

# Count stations per date distribution
print("\nYearly sampling patterns:")
print(regime_analysis)

print("\n[Historical patterns]")
stations_per_date = river_all.groupby('tanggal')['stasiun'].count()

# Filter by the date index (not by river_all's boolean mask)
pre_2022_dates = stations_per_date[stations_per_date.index.year < 2022]
post_2022_dates = stations_per_date[stations_per_date.index.year >= 2022]

print(f"  Pre-2022: ~{pre_2022_dates.median():.0f} stations/date (median)")
print(f"  2022+:    ~{post_2022_dates.median():.0f} stations/date (median)")

# %% [markdown]
#### 1.3 Missing Value Audit

# %%
print("\n[1.3] Missing Value Audit by Year")
print("-" * 70)

missing_by_year = river_all.groupby(river_all['tanggal'].dt.year)[value_cols].apply(
    lambda g: (g.isna().sum() / len(g) * 100).round(1)
)

print("\nMissing % by year (showing key parameters):")
key_params = ['lead', 'mbas_detergent', 'total_dissolved_solids', 'ph', 
              'biological_oxygen_demand', 'fecal_coliform']
print(missing_by_year[key_params])

# %% [markdown]
### Phase 2 - Create Daily, Long-format Dataset

# %%
print("\n" + "="*70)
print("PHASE 2: DAILY DATASET CREATION")
print("="*70)

# %% [markdown]
#### 2.1 Aggregate Multiple Measurements (Median)

# %%
print("\n[2.1] Aggregating multiple measurements per date-station...")

# Separate pH from other parameters
ph_param = 'ph'
other_params = [col for col in value_cols if col != ph_param]

# Group by date and station
grouped = river_all.groupby(['tanggal', 'stasiun'])

# Aggregate: median for most, mean for pH
aggregated_data = []

for (date, station), group in grouped:
    record = {'tanggal': date, 'stasiun': station}
    
    # Use median for most parameters (robust to outliers)
    for param in other_params:
        if param in group.columns:
            record[param] = group[param].median()
    
    # Use mean for pH
    if ph_param in group.columns:
        record[ph_param] = group[ph_param].mean()
    
    aggregated_data.append(record)

river_aggregated = pd.DataFrame(aggregated_data)

print(f"  → Before aggregation: {len(river_all)} rows")
print(f"  → After aggregation:  {len(river_aggregated)} rows")
print(f"  → Reduction: {(1 - len(river_aggregated)/len(river_all))*100:.1f}%")

# %% [markdown]
#### 2.2 Detect and Handle Outliers

# %%
print("\n[2.2] Outlier Detection & Capping...")

# Define reasonable upper bounds per parameter (based on water quality standards)
param_bounds = {
    'fecal_coliform': 1e8,     # CFU/100ml
    'total_coliform': 1e8,     # CFU/100ml
    'biological_oxygen_demand': 500,    # mg/L
    'chemical_oxygen_demand': 1000,     # mg/L
    'total_suspended_solids': 1000,     # mg/L
    'total_dissolved_solids': 5000,     # mg/L
    'chromium_vi': 200,        # µg/L
    'cadmium': 50,             # µg/L
    'copper': 5000,            # µg/L
    'lead': 500,               # µg/L
    'mercury': 50,             # µg/L
    'zinc': 10000,             # µg/L
    'oil_and_grease': 100,     # mg/L
    'mbas_detergent': 50,      # mg/L
    'ph': 14,                  # pH scale
}

# Cap outliers
outliers_detected = {}
for param, upper_bound in param_bounds.items():
    if param in river_aggregated.columns:
        original_max = river_aggregated[param].max()
        outlier_mask = river_aggregated[param] > upper_bound
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            river_aggregated.loc[outlier_mask, param] = upper_bound
            outliers_detected[param] = (outlier_count, original_max, upper_bound)

if outliers_detected:
    print(f"\n  Outliers detected and capped:")
    for param, (count, orig_max, cap) in outliers_detected.items():
        print(f"    {param}: {count} values capped from max={orig_max:.2f} to {cap}")
else:
    print("  ✓ No extreme outliers detected")

# %% [markdown]
#### 2.3 Reindex to Daily & Forward-Fill with Staleness Tracking

# %%
print("\n[2.3] Creating daily time series with staleness tracking...")

# Create daily date range for all stations
# Use actual data range for realistic expansion
start_date = river_aggregated['tanggal'].min()
end_date = pd.Timestamp('2025-12-31')  # Extend to end of 2025 for forecasting

date_range = pd.date_range(start=start_date, end=end_date, freq='D')
stations = sorted(river_aggregated['stasiun'].unique())

# Create daily skeleton
daily_skeleton = pd.DataFrame([
    {'tanggal': date, 'stasiun': station}
    for date in date_range
    for station in stations
])

print(f"  → Daily skeleton: {len(daily_skeleton):,} rows ({len(date_range)} days × {len(stations)} stations)")
print(f"  → Date range: {start_date.date()} to {end_date.date()}")

# Merge with aggregated data
river_daily = daily_skeleton.merge(
    river_aggregated, 
    on=['tanggal', 'stasiun'], 
    how='left'
)

print(f"  → Merged with actual data")

# %% [markdown]
#### 2.4 Forward-Fill and Calculate Staleness

# %%
print("\n[2.4] Forward-filling with staleness features...")

# Process each station separately
daily_expanded_list = []

for station in stations:
    station_data = river_daily[river_daily['stasiun'] == station].copy()
    station_data = station_data.sort_values('tanggal').reset_index(drop=True)
    
    # For each parameter, forward-fill and track staleness
    for param in value_cols:
        if param in station_data.columns:
            # Create staleness column name
            staleness_col = f'{param}_days_stale'
            
            # Mark rows with real measurements
            has_real_value = station_data[param].notna()
            
            # Forward fill the parameter
            station_data[param] = station_data[param].ffill()
            
            # Calculate days since last real measurement
            # Initialize staleness
            staleness = np.zeros(len(station_data), dtype=int)
            days_since = 0
            
            for i in range(len(station_data)):
                if has_real_value.iloc[i]:
                    days_since = 0  # Reset on real measurement
                else:
                    days_since += 1
                staleness[i] = days_since
            
            station_data[staleness_col] = staleness
    
    daily_expanded_list.append(station_data)

river_daily_expanded = pd.concat(daily_expanded_list, ignore_index=True)

print(f"  ✓ Created daily dataset: {river_daily_expanded.shape}")
print(f"  → Date range: {river_daily_expanded['tanggal'].min().date()} to {river_daily_expanded['tanggal'].max().date()}")

# %% [markdown]
#### 2.5 Add Availability Flags

# %%
print("\n[2.5] Adding parameter availability flags...")

# For each parameter, mark when it becomes available per station
for param in value_cols:
    flag_col = f'{param}_available'
    
    # Initialize flag as 0
    river_daily_expanded[flag_col] = 0
    
    # For each station, mark available after first non-null measurement
    for station in stations:
        station_mask = river_daily_expanded['stasiun'] == station
        station_data = river_daily_expanded[station_mask]
        
        # Find first valid measurement
        param_col = river_aggregated[river_aggregated['stasiun'] == station][param]
        if param_col.notna().any():
            first_valid_date = river_aggregated[
                (river_aggregated['stasiun'] == station) & 
                (river_aggregated[param].notna())
            ]['tanggal'].min()
            
            # Mark as available from that date onward
            river_daily_expanded.loc[
                station_mask & (river_daily_expanded['tanggal'] >= first_valid_date), 
                flag_col
            ] = 1

print(f"  ✓ Added {len(value_cols)} availability flags")

# %% [markdown]
#### 2.6 Export Final Dataset

# %%
print("\n[2.6] Exporting final daily dataset...")

# Organize columns: metadata, values, staleness, flags
metadata_cols = ['tanggal', 'stasiun']
value_cols_sorted = sorted(value_cols)
staleness_cols = [f'{param}_days_stale' for param in value_cols_sorted]
flag_cols = [f'{param}_available' for param in value_cols_sorted]

# Reorder columns for better organization
final_column_order = metadata_cols + value_cols_sorted + staleness_cols + flag_cols
river_daily_expanded = river_daily_expanded[final_column_order]

# Export
output_file = 'data/cleaned/river_quality_2015-2025_daily.csv'
river_daily_expanded.to_csv(output_file, index=False)

print(f"  ✓ Saved to: {output_file}")
print(f"  → Shape: {river_daily_expanded.shape}")
print(f"  → Columns: {len(river_daily_expanded.columns)}")
print(f"    - Metadata: {len(metadata_cols)}")
print(f"    - Parameters: {len(value_cols_sorted)}")
print(f"    - Staleness indicators: {len(staleness_cols)}")
print(f"    - Availability flags: {len(flag_cols)}")

# %% [markdown]
### Final Summary

# %%
print("\n" + "="*70)
print("DATA EXPANSION COMPLETE")
print("="*70)

print(f"\nInput (sparse):")
print(f"  - Records: {len(river_all):,}")
print(f"  - Date range: {river_all['tanggal'].min().date()} to {river_all['tanggal'].max().date()}")
print(f"  - Unique dates: {river_all['tanggal'].nunique()}")

print(f"\nOutput (daily expanded):")
print(f"  - Records: {len(river_daily_expanded):,}")
print(f"  - Date range: {river_daily_expanded['tanggal'].min().date()} to {river_daily_expanded['tanggal'].max().date()}")
print(f"  - Daily coverage: 100% (all days for all stations)")

print(f"\nKey features:")
print(f"  ✓ Multiple measurements aggregated (median for robustness)")
print(f"  ✓ Outliers capped to reasonable bounds")
print(f"  ✓ Forward-filled to daily frequency")
print(f"  ✓ Staleness tracking (days since last real measurement)")
print(f"  ✓ Availability flags (parameter availability per station)")

print(f"\nData quality guidance:")
print(f"  - Use staleness indicators to weight model confidence")
print(f"  - Filter by availability flags for reliable predictions")
print(f"  - Consider higher uncertainty when staleness > 30 days")

print("="*70)

gc.collect()
