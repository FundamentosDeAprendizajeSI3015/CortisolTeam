# Visualizacion Interactiva - Crypto ML Project

Modulo para generar visualizaciones de EDA y clustering, y construir un dashboard interactivo con Dash.

## Resumen de entrega (version corta)

- Scripts del modulo: `app/visualize.py`, `app/dash_app.py`
- Salida principal: dashboard Dash en `visualization/app/dash_app.py` y figuras en `visualization/outputs/`
- Datos de entrada: `data/crypto_raw.csv`, `data/features_clustering.csv`, `data/cluster_labels.csv`
- Fuente supervisado: `supervised/reports/supervised_report.md`

## Dashboard

- Dashboard interactivo en Dash con secciones de EDA, No supervisado y Supervisado.
- Incluye filtros por simbolo, rango de fechas, algoritmo y metrica.
- Visualiza comparaciones y metricas de modelos con graficas Plotly.

## Requisitos

Instalar dependencias desde la raiz del proyecto:

```bash
pip install -r requirements.txt
```

Requerimientos:

- kagglehub>=0.3.0
- ta>=0.11.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- plotly>=5.15.0
- dash>=2.15.0
- kaleido>=0.2.1
- joblib>=1.3.0
- scipy>=1.11.0
- pyarrow>=13.0.0
- reportlab>=4.0.0

## Ejecucion

### Dashboard interactivo (Dash)

```bash
python visualization/app/dash_app.py
```


### Archivos del modulo

- `app/visualize.py`: genera las figuras PNG y valida datos de entrada.
- `app/dash_app.py`: dashboard interactivo con Dash (EDA, No supervisado y Supervisado).
- `app/assets/style.css`: estilos del dashboard.
- `README.md`: documentacion consolidada del modulo.


### Uso de IA
Esta visualización fue desarrollada con apoyo de herramientas de inteligencia artificial (Claude, Anthropic).
