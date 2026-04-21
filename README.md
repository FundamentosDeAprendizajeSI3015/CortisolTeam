# [03] Unsupervised — Crypto ML Project

## Objetivo
Descubrir agrupaciones naturales entre criptomonedas usando clustering.
Incluye EDA enfocado en la preparacion de features para clustering.

## Estructura

```
├── data/
│   ├── crypto_raw.csv              # salida de load-data
│   ├── features_clustering.csv     # features normalizados por moneda
│   └── cluster_labels.csv          # etiquetas de cada algoritmo
├── unsupervised/
│   ├── eda_clustering.py           # feature engineering + normalizacion
│   ├── clustering.py               # K-Means, DBSCAN, Agglomerative
│   └── reports/
│       ├── k_selection.png         # codo + silhouette + davies-bouldin
│       ├── clusters_pca.png        # visualizacion PCA 2D por algoritmo
│       └── conclusions.txt         # conclusiones e interpretacion de resultados
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Uso

Correr en orden:

```bash
python unsupervised/eda_clustering.py
python unsupervised/clustering.py
```

## Features usados para clustering

| Feature | Descripcion |
|---------|-------------|
| mean_return | Retorno diario promedio |
| volatility | Volatilidad rolling 30d |
| sharpe_ratio | Retorno ajustado por riesgo (anualizado) |
| max_drawdown | Peor caida desde maximo historico |
| avg_volume | Volumen promedio de trading |

## Algoritmos

- **K-Means** — seleccion de K via metodo del codo, silhouette y Davies-Bouldin
- **DBSCAN** — deteccion de anomalias y monedas atipicas
- **Agglomerative** — clustering jerarquico con el mismo K optimo

## Resultados

- K optimo: 2 (silhouette=0.6726) — XEM es outlier claro en todos los metodos
- K=4 revela stablecoins (USDT, USDC) como grupo propio
- DBSCAN separa DeFi/nuevos de altcoins establecidos; BTC, ETH y DOGE son anomalias
- Ver `unsupervised/reports/conclusions.txt` para analisis completo

## Dependencias

`feature/load-data` — requiere `data/crypto_raw.csv`
