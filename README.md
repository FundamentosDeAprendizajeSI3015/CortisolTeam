# Crypto ML Project

Proyecto de Machine Learning sobre criptomonedas. Abarca desde la ingesta de datos hasta modelos supervisados, clustering y visualizacion interactiva.

**Dataset:** Cryptocurrency Historical Prices (Kaggle) — 23 monedas, 2013-2021.

---

## Fases del proyecto

| # | Rama | Carpeta | Estado |
|---|------|---------|--------|
| 01 | `feature/load-data` | `load_data/` | Completado |
| 02 | `feature/eda` | `eda/` | Completado |
| 03 | `feature/unsupervised` | `unsupervised/` | Completado |
| 04 | `feature/supervised` | `supervised/` | En progreso |
| 05 | `feature/visualization` | `visualization/` | Pendiente |
| 06 | `feature/scoring` | `scoring/` | Pendiente |

---

## Estructura del repositorio

```
├── data/                       # datasets generados por cada fase
├── load_data/                  # ingesta y validacion del dataset
├── eda/                        # analisis exploratorio
├── unsupervised/               # clustering (K-Means, DBSCAN, Agglomerative)
├── supervised/                 # clasificacion y regresion (en progreso)
├── visualization/              # dashboard interactivo (pendiente)
├── scoring/                    # evaluacion final y backtesting (pendiente)
└── requirements.txt
```

---

## Setup general

```bash
pip install -r requirements.txt
```

Cada fase tiene su propia carpeta con un script principal y un README con instrucciones especificas.

---

## Flujo de ramas

```
main
└── desarrollo          <- rama de integracion
    ├── feature/load-data
    ├── feature/eda
    ├── feature/unsupervised
    ├── feature/supervised
    ├── feature/visualization
    └── feature/scoring
```

Las ramas `feature/*` hacen merge a `desarrollo` una vez completadas.
