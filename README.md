# [01] Load Data — Crypto ML Project

## Objetivo
Descargar y validar el dataset Cryptocurrency Historical Prices (Kaggle),
produciendo `data/crypto_raw.csv` listo para la siguiente fase.

## Estructura

```
├── data/               # salida del script (ignorado el ZIP)
├── load_data/
│   └── load_data.py    # script principal
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Uso

```bash
python load_data/load_data.py
```

El script descarga el dataset desde Kaggle, concatena todos los CSVs por moneda,
valida integridad (fechas, nulos, duplicados) y guarda el resultado.

## Entregable

`data/crypto_raw.csv` — dataset combinado, limpio y ordenado por Symbol + Date.
