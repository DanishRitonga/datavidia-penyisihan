# Data Documentation - Jakarta Air Quality Classification

**Project**: Air Quality Classification for Jakarta
**Last Updated**: January 30, 2026
**Data Period**: 2010-2025

---

## Overview

This project aims to classify air pollution levels in Jakarta using ISPU (Air Quality Index) data as the primary dataset, supported by weather, population, water quality, holidays, and vegetation index data.

**Target Variable**: `kategori` (Air Quality Category)

- BAIK (Good)
- SEDANG (Moderate)
- TIDAK SEHAT (Unhealthy)
- SANGAT TIDAK SEHAT (Very Unhealthy)

---

## 1. Main Dataset: ISPU (Air Quality Index)

**Location**: `data/ISPU/`

### Files Structure

- `indeks-standar-pencemaran-udara-(ispu)-tahun-2010-komponen-data.csv` (2010)
- `indeks-standar-pencemaran-udara-(ispu)-tahun-2011-komponen-data.csv` (2011)
- ... (yearly files from 2010-2022)
- `data-indeks-standar-pencemar-udara-(ispu)-di-provinsi-dki-jakarta-2023-komponen-data.csv` (2023)
- `data-indeks-standar-pencemar-udara-(ispu)-di-provinsi-dki-jakarta-komponen-data-2024.csv` (2024)
- `data-indeks-standar-pencemar-udara-(ispu)-di-provinsi-dki-jakarta-komponen-data-2025.csv` (2025)

### Data Schema

| Column         | Type   | Description                       | Example              |
| -------------- | ------ | --------------------------------- | -------------------- |
| `periode_data` | String | Period identifier (YYYYMM)        | "201901"             |
| `tanggal`      | Date   | Measurement date                  | "2019-01-15"         |
| `pm10`         | Float  | Particulate Matter ≤10μm          | 75.3                 |
| `so2`          | Float  | Sulfur Dioxide                    | 12.5                 |
| `co`           | Float  | Carbon Monoxide                   | 8.2                  |
| `o3`           | Float  | Ozone                             | 45.6                 |
| `no2`          | Float  | Nitrogen Dioxide                  | 32.1                 |
| `max`          | Float  | Maximum pollutant value           | 75.3                 |
| `critical`     | String | Which pollutant is critical       | "PM10"               |
| `categori`     | String | **TARGET** - Air quality category | "SEDANG"             |
| `lokasi_spku`  | String | Monitoring station location       | "DKI1", "DKI2", etc. |

### Monitoring Stations

| Code | Approx. Location   | Administrative District |
| ---- | ------------------ | ----------------------- |
| DKI1 | Bundaran HI Area   | Jakarta Pusat (Central) |
| DKI2 | Kelapa Gading Area | Jakarta Utara (North)   |
| DKI3 | Jagakarsa Area     | Jakarta Selatan (South) |
| DKI4 | Kebon Jeruk Area   | Jakarta Barat (West)    |
| DKI5 | Lubang Buaya Area  | Jakarta Timur (East)    |

### Data Quality Notes

- ✅ Complete temporal coverage (2010-2025)
- ⚠️ Dates may not be in chronological order (requires sorting)
- ⚠️ Some suspicious values detected (e.g., `co=0` on 2019-08-05)
- ✅ Multiple locations provide spatial variation
- **Estimated Records**: ~29,000 samples (5,840 days × 5 locations)

---

## 2. Supporting Dataset: Weather Data

**Location**: `data/cuaca-harian/`

### Files Structure

- `cuaca-harian-dki1-bundaranhi.csv`
- `cuaca-harian-dki2-kelapagading.csv`
- `cuaca-harian-dki3-jagakarsa.csv`
- `cuaca-harian-dki4-lubangbuaya.csv`
- `cuaca-harian-dki5-kebonjeruk.csv`

### Expected Schema (TBD - Needs Verification)

Typical weather data columns:

- `tanggal` (Date) - **Join Key**
- `temperature` (Temperature in °C)
- `humidity` (Humidity in %)
- `rainfall` (Rainfall in mm)
- `wind_speed` (Wind speed in m/s)
- `wind_direction` (Wind direction)
- Possibly: `pressure`, `cloud_cover`

### Location Mapping to ISPU

| Weather Station | ISPU Code | Rationale                            |
| --------------- | --------- | ------------------------------------ |
| Bundaran HI     | DKI1      | Central Jakarta business district    |
| Kelapa Gading   | DKI2      | North Jakarta residential/commercial |
| Jagakarsa       | DKI3      | South Jakarta residential            |
| Kebon Jeruk     | DKI4      | West Jakarta mixed area              |
| Lubang Buaya    | DKI5      | East Jakarta                         |

### Data Connection

- **Join Type**: Left join (ISPU as primary)
- **Join Keys**: `tanggal` + location mapping
- **Expected Impact**: Weather conditions (temperature, humidity, wind) strongly influence pollutant dispersion

---

## 3. Supporting Dataset: Population Data

**Location**: `data/jumlah-penduduk/`

### Files Structure

- `data-jumlah-penduduk-provinsi-dki-jakarta-berdasarkan-kelompok-usia-dan-jenis-kelamin-tahun-2013-2021-komponen-data.csv`

### Expected Schema (TBD - Needs Verification)

Typical columns:

- `tahun` (Year)
- `kelompok_usia` (Age group)
- `jenis_kelamin` (Gender)
- `jumlah_penduduk` (Population count)
- `kecamatan` / `kelurahan` (District)

### Data Connection

- **Join Type**: Lookup by location and time period
- **Aggregation**: May need to aggregate by district to match ISPU locations
- **Expected Impact**: Population density correlates with pollution sources (traffic, industrial activity)
- **Temporal Granularity**: Yearly data (less granular than daily ISPU)

---

## 4. Supporting Dataset: River Water Quality

**Location**: `data/kualitas-air-sungai/`

### Files Structure

- `data-kualitas-air-sungai-komponen-data.csv`

### Expected Schema (TBD - Needs Verification)

Typical columns:

- `tanggal` (Date)
- `lokasi` / `nama_sungai` (Location/River name)
- Water quality parameters (BOD, COD, pH, etc.)

### Data Connection

- **Join Type**: Join by date and location (requires location mapping)
- **Challenge**: River locations likely use different naming convention than DKI1-5
- **Expected Impact**: Environmental health indicator; industrial waste affects both water and air quality
- **Use Case**: Proxy for industrial activity in the area

---

## 5. Supporting Dataset: National Holidays

**Location**: `data/libur-nasional/`

### Files Structure

- `dataset-libur-nasional-dan-weekend.csv`

### Expected Schema (TBD - Needs Verification)

Typical columns:

- `tanggal` (Date) - **Join Key**
- `keterangan` (Holiday description)
- `jenis` (Type: National Holiday / Collective Leave / Weekend)

### Data Connection

- **Join Type**: Left join on date
- **Feature Engineering**: Create binary feature `is_holiday`
- **Expected Impact**: Reduced traffic and industrial activity during holidays typically improves air quality
- **High Value**: Easy to integrate, clear causal relationship

---

## 6. Supporting Dataset: Vegetation Index (NDVI)

**Location**: `data/NDVI (vegetation index)/`

### Files Structure

- `indeks-ndvi-jakarta.csv`

### Expected Schema (TBD - Needs Verification)

Typical columns:

- `tanggal` or `periode` (Date/Period)
- `lokasi` (Location)
- `ndvi_value` (Vegetation index value, range: -1 to 1)

### Data Connection

- **Join Type**: Join by date/period and location
- **Challenge**: Location format needs mapping to DKI1-5
- **Expected Impact**: Green spaces absorb pollutants and produce oxygen; negative correlation with pollution
- **Temporal Granularity**: Possibly monthly or seasonal (satellite data)

---

## 7. Sample Submission

**Location**: `data/sample_submission.csv`

### Purpose

- Template for competition submission format
- Defines required output structure

### Expected Schema

- May contain sample IDs or date-location combinations
- Format for prediction submission

---

## Data Integration Strategy

### Priority 1: Core Features (Days 1-4)

1. **ISPU Data**
   - Temporal features (day, month, season, year)
   - Lag features (1, 7, 30 days)
   - Rolling statistics (7-day mean/std)
   - Pollutant interactions

2. **Holiday Data**
   - Binary feature: is_holiday
   - Day of week: is_weekend

### Priority 2: High-Impact Features (Days 5-6)

3. **Weather Data**
   - Temperature, humidity, rainfall
   - Wind speed and direction
   - Lagged weather (previous day)

### Priority 3: If Time Permits (Days 7-8)

4. **Population Data**
   - Population density by location
   - Demographic features

5. **NDVI Data**
   - Vegetation coverage by location
   - Seasonal vegetation changes

6. **River Water Quality**
   - Water quality indicators
   - Industrial activity proxy

---

## Data Quality Checklist

### Before Modeling

- [ ] Check for missing values in ISPU data
- [ ] Verify date formats across all datasets
- [ ] Create and validate location mappings
- [ ] Check for duplicate records
- [ ] Identify and handle outliers
- [ ] Verify class balance in target variable
- [ ] Ensure temporal alignment across datasets
- [ ] Document all data transformations

### Known Issues to Address

1. **ISPU Data**: Suspicious zero values (e.g., `co=0`)
2. **Date Ordering**: Files not in chronological order
3. **Location Formats**: Different naming conventions across datasets
4. **Temporal Granularity**: Mixing daily, monthly, yearly data
5. **Missing Data**: Need to verify completeness of weather data

---

## Estimated Dataset Sizes

| Dataset       | Estimated Records | Temporal Coverage  | Spatial Coverage   |
| ------------- | ----------------- | ------------------ | ------------------ |
| ISPU          | ~29,000           | Daily (2010-2025)  | 5 locations        |
| Weather       | ~29,000           | Daily              | 5 stations         |
| Population    | ~45               | Yearly (2013-2021) | Multiple districts |
| Holidays      | ~5,840            | Daily (2010-2025)  | Jakarta-wide       |
| River Quality | Unknown           | Varies             | Multiple rivers    |
| NDVI          | Unknown           | Monthly/Seasonal   | Jakarta-wide       |

---

## Feature Engineering Plan

### Temporal Features

- `year`, `month`, `day_of_week`, `quarter`
- `is_weekend`, `is_holiday`
- `season` (dry season vs rainy season)

### Lag Features (Time Series)

- `pm10_lag1`, `pm10_lag7`, `pm10_lag30`
- `o3_lag1`, `o3_lag7`
- Weather lags (temperature, humidity)

### Rolling Statistics

- `pm10_roll7_mean`, `pm10_roll7_std`
- `o3_roll7_mean`, `o3_roll7_std`

### Interaction Features

- `pm10_o3_ratio`
- `total_pollutants` (sum of all pollutants)
- `pm10_temperature_interaction`
- `humidity_wind_interaction`

### Spatial Features (Optional)

- `location` (categorical: DKI1-5)
- `neighbor_avg_pm10` (average from nearby stations)
- `population_density`
- `ndvi_value`

---

## Notes

- **Modeling Approach**: Gradient Boosting (LightGBM primary, XGBoost/CatBoost for ensemble)
- **Validation Strategy**: Time Series Split (respect temporal order)
- **Target Metric**: Accuracy and F1-Score (macro)
- **Expected Performance**: 75-85% accuracy
- **Time Constraint**: 10-day sprint
- **Hardware**: NVIDIA 3050 Laptop (prioritize CPU-based models)

---

## Next Steps

1. ✅ Document data structure (this file)
2. [ ] Load and explore ISPU data
3. [ ] Verify schemas of all supporting datasets
4. [ ] Create location mapping tables
5. [ ] Implement data loading pipeline
6. [ ] Perform EDA and visualizations
7. [ ] Build feature engineering pipeline
8. [ ] Train baseline model
9. [ ] Implement ensemble strategy
10. [ ] Evaluate and iterate
