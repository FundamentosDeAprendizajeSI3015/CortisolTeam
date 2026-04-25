# Crypto ML Project

Proyecto de Machine Learning sobre criptomonedas. Abarca desde la ingesta de datos hasta modelos supervisados, clustering y visualización interactiva.

**Dataset:** Cryptocurrency Historical Prices (Kaggle) — 23 monedas, 2013-2021.

---

## Fases del proyecto

| # | Rama | Carpeta | Estado |
|---|------|---------|--------|
| 01 | `feature/load-data` | `load_data/` | ✅ Completado |
| 02 | `feature/eda` | `eda/` | ✅ Completado |
| 03 | `feature/unsupervised` | `unsupervised/` | ✅ Completado |
| 04 | `feature/supervised` | `src/` | ✅ Completado |
| 05 | `feature/visualization` | `visualization/` | Pendiente |
| 06 | `feature/index_score` | `src/scoring.py` | ✅ Completado |

---

## Estructura del repositorio

```
├── data/                       # datasets generados por cada fase
│   ├── crypto_raw.csv          # dataset original combinado
│   ├── processed/
│   │   └── features.parquet    # features con indicadores técnicos
│   ├── features_clustering.csv # features para clustering
│   └── cluster_labels.csv      # etiquetas de clustering
├── load_data/                  # ingesta y validación del dataset
├── src/                        # scripts principales
│   ├── feature_engineering.py  # pipeline de features
│   ├── supervised.py           # clasificación y regresión
│   ├── eda.py                  # análisis exploratorio
│   └── scoring.py              # evaluación final y backtesting
├── unsupervised/               # clustering (K-Means, DBSCAN, Agglomerative)
├── notebooks/
│   └── 06_scoring.ipynb        # notebook de scoring documentado
├── models/                     # modelos serializados (.pkl)
├── reports/                    # reportes y figuras
│   ├── metrics_summary.csv     # métricas consolidadas
│   ├── model_report.pdf        # reporte PDF final
│   └── figures/                # 23 figuras generadas
└── requirements.txt
```

---

## Setup general

```bash
pip install -r requirements.txt
```

Cada fase tiene su propia carpeta con un script principal y un README con instrucciones específicas.

---

## Flujo de ramas

```
main
└── desarrollo          <- rama de integración
    ├── feature/load-data
    ├── feature/eda
    ├── feature/unsupervised
    ├── feature/supervised
    ├── feature/visualization
    └── feature/index_score   ← scoring (completado)
```

Las ramas `feature/*` hacen merge a `desarrollo` una vez completadas.
