# Jakarta Air Quality Prediction Project

Competition: Datavidia Penyisihan Round

## 📊 Overview

This project predicts air quality categories (BAIK, SEDANG, TIDAK SEHAT) for Jakarta's 5 monitoring stations (DKI1-DKI5) during September-November 2025.

## 🎯 Best Approach: Ensemble Classifier

**Current Best Submission:** `submission_ensemble_classifier.csv`
- **Distribution:** 329 SEDANG (72.3%), 112 TIDAK SEHAT (24.6%), 14 BAIK (3.1%)
- **Method:** XGBoost + LightGBM + CatBoost ensemble classifier
- **Expected Accuracy:** 35-45% (based on similar Colab notebook that achieved 40%)

### Key Improvements Over Regression Approach:
1. ✅ Direct classification (not regression → ISPU → category)
2. ✅ 3-model ensemble with 95.5% validation accuracy
3. ✅ Station-specific patterns (DKI4 identified as worst station)
4. ✅ Risk-averse thresholds for air quality safety
5. ✅ Uses forecasted pollutant values from regression models

## 📁 Project Structure

### Main Prediction Scripts

```
forecasting_ensemble_classifier.py  [★ RECOMMENDED]
├─ Ensemble classifier approach (XGBoost + LightGBM + CatBoost)
├─ Uses forecasted pollutants as input
├─ Outputs: submission_ensemble_classifier.csv
└─ Expected: 35-45% accuracy

forecasting_kategori_rulebased.py  [Alternative]
├─ Rule-based ISPU threshold approach
├─ Uses forecasting.py output
├─ Outputs: submission.csv (28% accuracy - previous attempt)
└─ Distribution: 73% SEDANG, 25% TIDAK SEHAT

forecasting.py  [Base Model]
├─ XGBoost regression for 6 pollutants
├─ 30 pruned features, R² 0.84-0.97
├─ Applies station corrections and seasonal adjustments
└─ Outputs: forecasting_predictions_2025-09-to-11.csv
```

### Data Processing Scripts

```
ISPU.py, NDVI.py, population.py, RIVER.py, WEATHER.py
├─ Load and clean raw data sources
└─ Merge into unified ISPU_2010-2024.csv

*-FE.py (Feature Engineering variants)
├─ Advanced feature engineering for each data source
└─ Used during model development
```

### Input Data Files

```
data/
├─ sample_submission.csv          [Competition template]
├─ ISPU/                          [Air quality 2010-2024]
├─ cuaca-harian/                  [Weather data]
├─ kualitas-air-sungai/           [River quality]
├─ NDVI (vegetation index)/       [Vegetation index]
├─ jumlah-penduduk/               [Population data]
└─ libur-nasional/                [Holiday calendar]

ISPU_2010-2024.csv                [Merged training data: 14,725 samples]
```

### Output Files

```
submission_ensemble_classifier.csv  [★ Best submission - Try on Kaggle!]
submission.csv                      [Previous submission - 28% accuracy]

forecasting_predictions_2025-09-to-11.csv
├─ Pollutant forecasts (455 samples: 91 days × 5 stations)
└─ Columns: pm_sepuluh, pm_duakomalima, sulfur_dioksida, karbon_monoksida, ozon, nitrogen_dioksida

forecasting_predictions_with_kategori_2025-09-to-11.csv
└─ Predictions + ISPU categories + critical pollutant
```

### Documentation

```
FEATURE_PRUNING_REPORT.md         [Feature selection analysis: 171→30 features]
data.md                           [Data documentation]
catboost_info/                    [CatBoost training logs]
```

## 🚀 How to Run

### Quick Start: Generate Best Submission

```bash
# 1. Generate pollutant forecasts (base regression model)
python forecasting.py

# 2. Generate final category predictions (ensemble classifier)
python forecasting_ensemble_classifier.py

# Output: submission_ensemble_classifier.csv
```

### Alternative: Rule-Based Approach

```bash
# 1. Generate pollutant forecasts
python forecasting.py

# 2. Apply ISPU rule-based thresholds
python forecasting_kategori_rulebased.py

# Output: submission.csv
```

### Rebuild Training Data (if needed)

```bash
python ISPU.py       # Process air quality data
python WEATHER.py    # Process weather data  
python RIVER.py      # Process river quality data
python NDVI.py       # Process vegetation index
python population.py # Process population data
```

## 📈 Model Architecture

### Ensemble Classifier Approach (Recommended)

```
Training Data (Sept-Nov 2010-2024)
├─ 3,442 samples
├─ 69% SEDANG, 22% TIDAK SEHAT, 8% BAIK
└─ 18 elite features

↓ [Feature Engineering]
├─ 3-day rolling memory features
├─ Day-of-year, weekend, holiday indicators
├─ Station one-hot encoding (DKI1-DKI5)
└─ PM2.5 calculated from PM10 (ratio 1.46)

↓ [Ensemble Training]
├─ XGBoost Classifier (95.8% val accuracy)
├─ LightGBM Classifier (95.4% val accuracy)  
└─ CatBoost Classifier (95.2% val accuracy)

↓ [Prediction Calibration]
├─ Average ensemble probabilities
├─ Station multipliers (DKI4: 1.8×, DKI2: 1.6×)
├─ Month multipliers (Nov: 1.2×)
└─ Risk-averse threshold (TIDAK SEHAT > 8%)

↓ [Output]
submission_ensemble_classifier.csv
├─ 455 predictions
├─ 72% SEDANG, 25% TIDAK SEHAT, 3% BAIK
└─ DKI4 has 45 unhealthy days (worst station)
```

### Regression → Rule-Based Approach (Previous)

```
Training Data (2010-2024)
├─ 14,725 samples
└─ 30 features (after pruning from 171)

↓ [XGBoost Regression - 6 models]
├─ pm_sepuluh, pm_duakomalima, sulfur_dioksida
├─ karbon_monoksida, ozon, nitrogen_dioksida
└─ R² 0.84-0.97 per pollutant

↓ [Post-Processing]
├─ Station bias corrections
├─ Seasonal adjustments (rainy season transition)
├─ Global corrections (PM: 0.535×, CO: 0.55×)
└─ Temporal variation (month-specific volatility)

↓ [ISPU Rule-Based Classification]
├─ Calculate ISPU for each pollutant
├─ Worst pollutant determines category
└─ Map to BAIK/SEDANG/TIDAK SEHAT

↓ [Output]  
submission.csv (28% accuracy)
```

## 🔍 Key Findings

### Historical Patterns (Sept-Nov 2010-2024)
- **Category Distribution:** 69% SEDANG, 20% TIDAK SEHAT, 8% BAIK
- **Critical Pollutant:** PM2.5 dominates (not CO)
- **Worst Station:** DKI4 (Lubang Buaya) consistently worst
- **Seasonal Trend:** November is worse (dry season peak)

### Model Insights
- **Validation Accuracy:** 95.5% on historical data
- **Station Ranking:** DKI4 > DKI2 > DKI5 > DKI1 ≈ DKI3
- **Top Features:** PM2.5, PM10, Ozone, 3-day memory features
- **Kaggle Results:** 
  - Regression approach: 28% accuracy
  - Ensemble classifier: Expected 35-45% (untested)

## 📝 Submission Files

| File | Distribution | Method | Status |
|------|-------------|--------|--------|
| `submission_ensemble_classifier.csv` | 72% SEDANG, 25% TIDAK SEHAT | Ensemble Classifier | ★ Try this! |
| `submission.csv` | 73% SEDANG, 25% TIDAK SEHAT | Regression + Rules | 28% accuracy |

## 🛠️ Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost
```

## 📚 References

- **Colab Notebook:** Achieved 40% accuracy with similar ensemble approach
- **IQAir Jakarta:** Real-world validation of seasonal patterns
- **ISPU Thresholds:** Indonesian Air Quality Index standards

## 👥 Team

Datavidia Penyisihan Round Submission

Last Updated: February 8, 2026
