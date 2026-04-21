# Crypto ML Project

Pipeline de machine learning sobre el dataset Cryptocurrency Historical Prices
(Kaggle: sudalairajkumar/cryptocurrencypricehistory) con 23 criptomonedas y datos diarios
desde 2013 hasta 2021.

---

## Estructura del proyecto

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

## Ramas

| Rama | Contenido |
|------|-----------|
| `feature/load-data` | Descarga y validación del dataset |
| `feature/eda` | Análisis Exploratorio (EDA) |
| `feature/supervised` | Feature engineering y modelos supervisados |

## Setup rápido

```bash
pip install pandas pyarrow numpy matplotlib seaborn plotly \
            statsmodels scikit-learn xgboost ta joblib jupyter
```

## Uso

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
