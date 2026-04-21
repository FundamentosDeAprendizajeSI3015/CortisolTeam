# Crypto ML Project

Machine learning pipeline using the Cryptocurrency Historical Prices dataset (Kaggle: sudalairajkumar/cryptocurrencypricehistory), featuring 23 cryptocurrencies with daily data from 2013 to 2021

---

## Project Structure 

```
├── data/
│   ├── crypto_raw.csv          # Dataset crudo validado (load-data)
│   └── processed/              # Features procesados (supervised)
├── load_data/
│   └── load_data.py            # [01] Descarga, validación e ingesta
├── notebooks/
│   ├── 02_eda.ipynb            # [02] Análisis Exploratorio de Datos
│   └── 04_supervised.ipynb     # [04] Modelos Supervisados
├── reports/
│   ├── figures/                # Figuras exportadas (PNG)
│   └── eda_summary.md          # Resumen de hallazgos del EDA
├── src/
│   ├── feature_engineering.py  # Indicadores técnicos y target
│   └── supervised.py           # Entrenamiento y evaluación de modelos
├── models/                     # Modelos serializados (.pkl)
└── requirements.txt
```

## Branches

| Branch | Content |
|------|-----------|
| `feature/load-data` | Dataset download and schema validation |
| `feature/eda` | Exploratory Data Analysis (EDA) |
| `feature/supervised` | Feature engineering and supervised learning models |

## Quick Setup

```bash
pip install pandas pyarrow numpy matplotlib seaborn plotly \
            statsmodels scikit-learn xgboost ta joblib jupyter
```

## Use

```bash
# [01] Cargar datos
python load_data/load_data.py

# [02] EDA
jupyter notebook notebooks/02_eda.ipynb

# [03] Features + modelos supervisados
python src/feature_engineering.py
python src/supervised.py
jupyter notebook notebooks/04_supervised.ipynb
```
