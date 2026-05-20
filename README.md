# Crypto ML Project — CortisolTeam

Pipeline completo de Machine Learning aplicado a criptomonedas: desde ingesta y generación sintética hasta clustering, modelos supervisados, SVM-MKL y backtesting.

**Dataset:** 23 monedas reales (Kaggle, 2013–2021) + 220 monedas sintéticas (GBM + Jump-Diffusion) = **~988K filas totales**.

## Equipo

- **Pablo Cabrejos Munera**
- **Juan Manuel Florez Robledo**
- **Jean Carlo Londoño Ocampo**
- **Alejandro Garces Ramirez**
- **Paula Inés Llanos López**

---

## Fases del proyecto

| # | Carpeta | Descripción | Estado |
|---|---------|-------------|--------|
| 01 | `load_data/` | Ingesta de datos reales (Kaggle) | ✅ |
| 01b | `load_data/` | Generación sintética con IA (SYN001–SYN220) | ✅ |
| 02 | `eda/` | Análisis exploratorio (Matplotlib + Seaborn) | ✅ |
| 03 | `unsupervised/` | Clustering (K-Means, DBSCAN, Agglomerative, One-Class SVM) | ✅ |
| 04 | `supervised/` | Feature engineering + clasificación + regresión | ✅ |
| 05 | `svm_kernels/` | SVM con Multiple Kernel Learning (MKL) | ✅ |
| 06 | `scoring/` | Evaluación final y backtesting | ✅ |
| 07 | `visualization/` | Dashboard HTML interactivo | ✅ |

---

## Requisitos del sistema
 
- **Python:** 3.9 o superior
- **Sistema operativo:** Compatible con Windows, macOS y Linux
- Se recomienda usar un entorno virtual (`venv` o `conda`)

---

## Setup

```bash
pip install -r requirements.txt
# SVM-MKL tiene dependencias adicionales:
pip install -r svm_kernels/requirements.txt
```

## Ejecución

```bash
python load_data/load_data.py
python load_data/synthetic_data.py
python eda/eda.py
python unsupervised/eda_clustering.py && python unsupervised/clustering.py && python unsupervised/svm_analysis.py
python supervised/feature_engineering.py && python supervised/supervised.py
python -m svm_kernels.svm_mkl          # ejecutar como módulo, no como script directo
python scoring/src/scoring.py
python visualization/app/main.py
```

---

## Estructura

```
├── data/                  # datasets intermedios y finales
├── load_data/             # ingesta y generación sintética
├── eda/                   # análisis exploratorio
├── unsupervised/          # clustering y anomaly detection
├── supervised/            # modelos supervisados y feature engineering
├── svm_kernels/           # SVM-MKL sobre datos cripto
├── scoring/               # métricas finales y backtesting
├── visualization/         # dashboard interactivo
└── requirements.txt
```

---

## Resultados clave

- **Clustering:** K-Means K=2 silhouette=0.91 · XEM outlier en 4/4 métodos
- **Clasificación:** Random Forest · Test AUC=0.591 · Test F1=0.65
- **Regresión:** XGBoost · Test R²=0.916 (cota superior, datos sintéticos)
- **SVM-MKL:** Anti-Natural MKL · AUC=0.528 · todos los MKL > kernel único

---
 
## Créditos
 
El módulo `svm_kernels/` está adaptado del repositorio [extremality_mkl](https://github.com/maospina1041/extremality_mkl) de **maospina1041**.
 
---

## Uso de IA
 
Durante el desarrollo de este proyecto utilizamos **Claude** (Anthropic) como herramienta de apoyo en inteligencia artificial.
