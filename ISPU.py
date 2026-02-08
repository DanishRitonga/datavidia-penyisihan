# %% [markdown]
# # ISPU Data Pipeline
# This script combines all ISPU data from 2010-2025 into a unified dataset

# %% Imports and Setup
import pandas as pd
import numpy as np
from pathlib import Path
import re

pd.set_option('display.max_columns', None)

# %% Configuration
# Define paths
ispu_folder = Path("data/ISPU")
output_file = "data/cleaned/ISPU_2010-2025.csv"

# Define station mapping
stasiun_mapping = {
    'DKI1': 'Jakarta Pusat',
    'DKI2': 'Jakarta Utara',
    'DKI3': 'Jakarta Selatan',
    'DKI4': 'Jakarta Timur',
    'DKI5': 'Jakarta Barat'
}

# %% Define Functions

def extract_station_code(station_str):
    """Extract DKIx code from station string"""
    if pd.isna(station_str):
        return None
    match = re.search(r'(DKI\d)', str(station_str))
    return match.group(1) if match else None

def fix_excel_dates(date_series):
    """Fix Excel numeric dates (e.g., 44926.625 -> actual date)"""
    def convert_date(val):
        if pd.isna(val):
            return val
        # If it's already a string date, return it
        if isinstance(val, str) and '-' in val:
            return val
        # If it's a numeric Excel date, convert it
        try:
            num_val = float(val)
            if num_val > 40000:  # Likely an Excel date (after year 2009)
                # Excel date starts from 1900-01-01, but with known bugs
                base_date = pd.Timestamp('1899-12-30')
                return (base_date + pd.Timedelta(days=num_val)).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            pass
        return val
    
    return date_series.apply(convert_date)

def standardize_format(df, year):
    """
    Standardize ISPU data format based on year.
    Handles all variations from 2010-2025.
    """
    print(f"  Processing {year}...")
    df = df.copy()
    
    # Year-specific preprocessing
    if year == 2022:
        # Fix Excel numeric dates
        df['tanggal'] = fix_excel_dates(df['tanggal'])
    
    # Build column mapping based on year
    if 2023 <= year <= 2025:
        # Already has correct column names
        column_mapping = {}
    elif year == 2022:
        # Has 'pm_10' with underscore, pm_duakomalima
        column_mapping = {
            'pm_10': 'pm_sepuluh',
            'so2': 'sulfur_dioksida',
            'co': 'karbon_monoksida',
            'o3': 'ozon',
            'no2': 'nitrogen_dioksida',
            'critical': 'parameter_pencemar_kritis',
            'categori': 'kategori',
            'lokasi_spku': 'stasiun'
        }
    elif year == 2021:
        # Has pm25 column
        column_mapping = {
            'pm10': 'pm_sepuluh',
            'pm25': 'pm_duakomalima',
            'so2': 'sulfur_dioksida',
            'co': 'karbon_monoksida',
            'o3': 'ozon',
            'no2': 'nitrogen_dioksida',
            'critical': 'parameter_pencemar_kritis',
            'categori': 'kategori'
        }
    elif 2011 <= year <= 2020:
        # Has lokasi_spku column
        column_mapping = {
            'pm10': 'pm_sepuluh',
            'so2': 'sulfur_dioksida',
            'co': 'karbon_monoksida',
            'o3': 'ozon',
            'no2': 'nitrogen_dioksida',
            'critical': 'parameter_pencemar_kritis',
            'categori': 'kategori',
            'lokasi_spku': 'stasiun'
        }
    else:  # 2010
        # Has stasiun column with full name
        column_mapping = {
            'pm10': 'pm_sepuluh',
            'so2': 'sulfur_dioksida',
            'co': 'karbon_monoksida',
            'o3': 'ozon',
            'no2': 'nitrogen_dioksida',
            'critical': 'parameter_pencemar_kritis',
            'categori': 'kategori'
        }
    
    # Apply column mapping
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Extract station code
    df['stasiun'] = df['stasiun'].apply(extract_station_code)
    
    # For 2024-2025: construct full date from periode_data and tanggal (day number)
    if 2024 <= year <= 2025 and 'bulan' in df.columns:
        if df['tanggal'].dtype in ['int64', 'float64']:
            df['year'] = df['periode_data'] // 100
            df['month'] = df['periode_data'] % 100
            df['day'] = df['tanggal'].astype(int)
            
            date_df = df[['year', 'month', 'day']].copy()
            df['tanggal'] = pd.to_datetime(date_df, errors='coerce').dt.strftime('%Y-%m-%d')
            
            df = df.drop(['year', 'month', 'day', 'bulan'], axis=1)
    
    # Handle missing values in pollution columns
    pollution_cols = ['pm_sepuluh', 'sulfur_dioksida', 'karbon_monoksida', 
                      'ozon', 'nitrogen_dioksida']
    if 'pm_duakomalima' in df.columns:
        pollution_cols.append('pm_duakomalima')
    
    for col in pollution_cols:
        if col in df.columns:
            df[col] = df[col].replace(['---', '-', ''], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def load_and_process_file(filepath):
    """Load a single ISPU file and standardize its format"""
    # Extract year from filename
    year_match = re.search(r'(\d{4})', filepath.name)
    if not year_match:
        print(f"  Warning: Could not extract year from {filepath.name}")
        return None
    
    year = int(year_match.group(1))
    
    # Load the file
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  Error loading {filepath.name}: {e}")
        return None
    
    # Standardize the format
    df = standardize_format(df, year)
    
    return df

def clean_dataframe(df):
    """Apply cleaning steps to the combined dataframe"""
    print("\nApplying cleaning steps...")
    
    # Remove rows with "TIDAK ADA DATA"
    original_len = len(df)
    df = df[df['kategori'] != 'TIDAK ADA DATA'].copy()
    print(f"  Removed {original_len - len(df)} rows with 'TIDAK ADA DATA'")
    
    # Remove rows with None/NaN station codes
    original_len = len(df)
    df = df[df['stasiun'].notna()].copy()
    if original_len > len(df):
        print(f"  Removed {original_len - len(df)} rows with invalid station codes")
    
    # Ensure tanggal is in proper format
    df['tanggal_datetime'] = pd.to_datetime(df['tanggal'], errors='coerce')
    df = df[df['tanggal_datetime'].notna()].copy()  # Remove rows with invalid dates
    df['tanggal'] = df['tanggal_datetime'].dt.strftime('%Y-%m-%d')
    
    # Sort by date and station
    df = df.sort_values(['tanggal_datetime', 'stasiun']).reset_index(drop=True)
    print(f"  Sorted by date and station")
    
    # Fill NULL values using forward fill within each station
    pollution_cols = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 
                      'ozon', 'nitrogen_dioksida']
    # Only fill columns that exist
    existing_cols = [col for col in pollution_cols if col in df.columns]
    df[existing_cols] = df.groupby('stasiun')[existing_cols].ffill()
    null_counts = df[existing_cols].isnull().sum()
    print(f"  Filled NULL values using forward fill")
    print(f"  Remaining NULL values:\n{null_counts}")
    
    # Handle duplicates by aggregating mean values for duplicate date-station combinations
    original_len = len(df)
    duplicated_mask = df.duplicated(subset=['tanggal', 'stasiun'], keep=False)
    n_duplicates = duplicated_mask.sum()
    
    if n_duplicates > 0:
        print(f"  Found {n_duplicates} rows with duplicate date-station combinations")
        print(f"  Aggregating by mean for numeric columns...")
        
        # Define pollution columns for mean aggregation
        pollution_cols = ['pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida', 
                          'ozon', 'nitrogen_dioksida']
        existing_pollution_cols = [col for col in pollution_cols if col in df.columns]
        
        # Build aggregation dictionary
        agg_dict = {}
        for col in df.columns:
            if col in ['tanggal', 'stasiun']:
                continue  # These are grouping keys
            elif col in existing_pollution_cols:
                agg_dict[col] = 'mean'  # Average pollution measurements
            elif col == 'periode_data':
                agg_dict[col] = 'first'
            else:
                agg_dict[col] = 'first'  # For other columns, keep first value
        
        # Group and aggregate
        df = df.groupby(['tanggal', 'stasiun'], as_index=False).agg(agg_dict)
        
        print(f"  Aggregated {n_duplicates} rows into {len(df)} unique date-station combinations")
        print(f"  Removed {original_len - len(df)} duplicate entries")
    else:
        print(f"  No duplicate date-station combinations found")
    
    # Create ID column
    df['ID'] = df['tanggal'] + '_' + df['stasiun']
    print(f"  Created ID column")

    # map to 'TIDAK SEHAT'
    df['kategori'] = df['kategori'].apply(
        lambda x: 'TIDAK SEHAT' if x in ['BERBAHAYA', 'SANGAT TIDAK SEHAT'] else x
    )
    
    # Drop temporary column
    df = df.drop('tanggal_datetime', axis=1)
    
    return df

def select_final_columns(df):
    """Select and order columns to match target format"""
    # Define target column order (keeping pm_duakomalima, dropping max and parameter_pencemar_kritis)
    target_columns = [
        'ID', 'periode_data', 'tanggal', 'stasiun',
        'pm_sepuluh', 'pm_duakomalima', 'sulfur_dioksida', 'karbon_monoksida',
        'ozon', 'nitrogen_dioksida', 'kategori'
    ]
    
    # Select only columns that exist
    available_columns = [col for col in target_columns if col in df.columns]
    df = df[available_columns].copy()
    
    return df

# %% Load and Combine All Files
print("="*60)
print("ISPU Data Pipeline - Combining 2010-2025 Data")
print("="*60)

# Get all ISPU CSV files
csv_files = sorted(ispu_folder.glob("*.csv"))
print(f"\nFound {len(csv_files)} CSV files in {ispu_folder}")

# Load and process each file
all_dataframes = []
for filepath in csv_files:
    print(f"\nProcessing: {filepath.name}")
    df = load_and_process_file(filepath)
    if df is not None and len(df) > 0:
        all_dataframes.append(df)
        print(f"  Loaded {len(df)} rows")

# Combine all dataframes
print("\n" + "="*60)
print("Combining all datasets...")
combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f"Total rows before cleaning: {len(combined_df)}")

# %% Clean Combined Data
cleaned_df = clean_dataframe(combined_df)

# %% Select Final Columns
final_df = select_final_columns(cleaned_df)

# %% Data Quality Report
print("\n" + "="*60)
print("DATA QUALITY REPORT")
print("="*60)

print(f"\nFinal dataset shape: {final_df.shape}")
print(f"Date range: {final_df['tanggal'].min()} to {final_df['tanggal'].max()}")
print(f"Unique stations: {sorted([s for s in final_df['stasiun'].unique() if s is not None])}")
print(f"Total unique IDs: {final_df['ID'].nunique()}")

# Check for duplicates
duplicate_ids = final_df[final_df.duplicated('ID', keep=False)]
if len(duplicate_ids) > 0:
    print(f"\nWARNING: Found {len(duplicate_ids)} duplicate IDs")
    print(duplicate_ids[['ID', 'tanggal', 'stasiun']].head(10))
else:
    print(f"\n✓ No duplicate IDs found")

# NULL value summary
print("\nNULL values by column:")
for col in final_df.columns:
    null_count = final_df[col].isnull().sum()
    null_pct = (null_count / len(final_df)) * 100
    print(f"  {col}: {null_count:,} ({null_pct:.1f}%)")

# Rows by year
print("\nRows by year:")
final_df['year'] = final_df['tanggal'].str[:4]
year_counts = final_df['year'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"  {year}: {count:,} rows")
final_df = final_df.drop('year', axis=1)

# %% Display Sample Data
print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)

print("\nFirst 10 rows:")
print(final_df.head(10).to_string())

print("\n\nLast 10 rows:")
print(final_df.tail(10).to_string())

# %% Save to CSV
print("\n" + "="*60)
print("SAVING DATA")
print("="*60)

# Create output directory if it doesn't exist
output_path = Path(output_file)
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save to CSV
final_df.to_csv(output_file, index=False)
print(f"\n✓ Data saved to: {output_file}")
print(f"  Total rows: {len(final_df):,}")
print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)
