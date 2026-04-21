# Resumen de Hallazgos — EDA Crypto ML Project

**Dataset:** Cryptocurrency Historical Prices (Kaggle: sudalairajkumar/cryptocurrencypricehistory)  
**Archivo fuente:** `data/crypto_raw.csv`  
**Período:** 2013-04-29 → 2021-07-06  
**Monedas:** 23 (AAVE, ADA, ATOM, BNB, BTC, CRO, DOGE, DOT, EOS, ETH, LINK, LTC, MIOTA, SOL, TRX, UNI, USDC, USDT, WBTC, XEM, XLM, XMR, XRP)  
**Registros totales:** 37,082

---

## 1. Calidad de datos

- **Sin valores nulos** en ninguna columna numérica — el pipeline de `load_data.py` ya los eliminó.
- **Sin duplicados** (Symbol, Date).
- Cobertura temporal muy desigual: BTC tiene 2,991 registros (desde 2013) mientras AAVE apenas 275 (desde 2020). Para modelos supervisados se recomienda usar **BTC, ETH, LTC, XRP y XMR** que tienen la mayor cobertura histórica.
- Las columnas disponibles son: `SNo`, `Name`, `Symbol`, `Date`, `High`, `Low`, `Open`, `Close`, `Volume`, `Marketcap`.

---

## 2. Estadísticas descriptivas de retornos diarios

| Moneda | Std (volatilidad) | Skewness | Kurtosis | Observación |
|--------|:-----------------:|:--------:|:--------:|-------------|
| XEM    | alto              | 16.04    | 257.4    | Evento extremo único |
| DOGE   | alto              | 37.12    | 1377.0   | Crashes extremos 2021 |
| BTC    | bajo-medio        | 0.24     | 10.27    | Más estable del grupo |
| ETH    | bajo-medio        | 0.86     | 7.85     | Segundo más estable |
| USDT   | muy bajo          | 11.72    | 1042.7   | Stablecoin |
| USDC   | muy bajo          | 0.58     | 19.4     | Stablecoin |

**Conclusión:** Ninguna criptomoneda sigue una distribución normal. Todas exhiben **leptocurtosis** (colas pesadas). Los QQ-plots confirman que los retornos extremos son mucho más frecuentes que lo que predice la distribución normal.

---

## 3. Series de tiempo y precios

- BTC alcanzó máximo histórico (en este dataset) de ~$58,000 en feb-2021.
- ETH y BNB mostraron crecimiento exponencial en 2020-2021, superando a BTC en términos de retorno porcentual desde su primer dato disponible.
- El **volumen** de BTC domina el mercado; en 2020-2021 se registraron los picos históricos.
- Precio normalizado (base=100): BNB fue el activo de mayor rendimiento relativo en el período, seguido de ETH.

---

## 4. Correlaciones entre monedas

**Pares más correlacionados (retornos diarios):**
1. WBTC–BTC: 0.93 — prácticamente el mismo activo (Wrapped Bitcoin)
2. WBTC–ETH: 0.74
3. WBTC–LTC: 0.71
4. UNI–AAVE: 0.69 — ambas DeFi tokens
5. ETH–EOS: 0.68

**Pares menos correlacionados:**
- USDT y USDC con cualquier criptomoneda no-stable: correlaciones cercanas a cero o negativas.
- XEM y DOGE muestran correlaciones bajas con el resto (eventos idiosincráticos dominantes).

**Implicación para modelos:** WBTC y BTC son redundantes como features — usar solo uno. Las stablecoins aportan poco como predictores de precio pero pueden ser útiles para detectar contexto de mercado.

---

## 5. Volatilidad rolling

- **Picos de volatilidad** identificados:
  - 2017-2018: bull run y corrección posterior
  - Mar-2020: crash COVID-19 (vol_30d de BTC llegó a ~8%)
  - Ene-Feb 2021: rally alcista con aumento de volatilidad
- La volatilidad de 90 días suaviza correctamente las señales de corto plazo.
- DOGE y altcoins pequeñas tienen volatilidad estructuralmente mayor que BTC y ETH.

---

## 6. Detección de outliers

| Moneda | Outliers Z-Score | % | Outliers IQR | % |
|--------|:----------------:|:-:|:------------:|:-:|
| DOGE   | ~15              |~6%| ~60          |~23%|
| XEM    | ~10              |~4%| ~50          |~20%|
| BTC    | ~8               |~0.3%|~100       |~4% |
| ETH    | ~6               |~0.3%|~90        |~4% |

- Z-Score detecta los eventos más extremos; IQR es más agresivo y captura más días "inusuales".
- Los outliers de BTC coinciden con eventos históricos identificables: crash COVID (mar-2020), rally 2017, crash 2018.
- **Decisión de ingeniería:** No eliminar outliers en el dataset; sí agregarlos como features binarias (`outlier_z`, `outlier_iqr`) para que el modelo aprenda el contexto de mercado extremo.

---

## 7. Distribución de retornos

- Todos los retornos muestran **asimetría positiva** (excepto ATOM con -0.09 y WBTC con -0.01).
- Colas derechas (retornos positivos extremos) más pesadas que las izquierdas en la mayoría.
- Los QQ-plots muestran desviación clara de la normal en las colas.
- **Implicación para modelos:** Modelos basados en supuesto de normalidad (ej: OLS puro) estarán mal calibrados. Preferable usar modelos robustos (Random Forest, XGBoost) o transformar retornos con `np.sign(r) * np.log1p(|r|)`.

---

## 8. Estacionalidad

- **Por mes:** En BTC, enero, abril y noviembre tienden a tener retornos promedio positivos. Febrero y septiembre tienden a ser negativos. El patrón no es consistente entre monedas.
- **Por día de semana:** Los criptoactivos operan 7 días. No hay "efecto lunes" claro como en mercados de acciones, pero algunos activos muestran retornos levemente más altos en fin de semana.
- **Conclusión:** La estacionalidad es débil. Usar variables de mes y día de semana como features dummy puede aportar señal marginal pero no será el driver principal de los modelos.

---

## Decisiones para la fase de Supervisado

1. **Monedas prioritarias:** BTC y ETH (mayor historia, menor ruido por outliers extremos). Modelar por separado o incluir Symbol como feature categórica.
2. **Features a crear:** SMA 7/14/30, EMA 14, RSI 14, Bollinger Bands, MACD, vol_30d, vol_90d, retorno lag-1, lag-2, lag-5.
3. **Target:** `Close(t+1) > Close(t)` → clasificación binaria con `target=1`.
4. **Split temporal:** 80/10/10 sin shuffle — los datos tienen estructura temporal y no se puede mezclar.
5. **WBTC vs BTC:** No incluir ambos en el mismo modelo — son colineales (r=0.93).
6. **Outliers:** No eliminar; incluir como features binarias.
7. **Stablecoins (USDT, USDC):** Excluir como targets de predicción de precio; pueden ser útiles como contexto.
