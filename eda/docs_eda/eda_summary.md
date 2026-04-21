# Findings Summary — EDA Crypto ML Project

**Dataset:** Cryptocurrency Historical Prices (Kaggle: sudalairajkumar/cryptocurrencypricehistory)  
**Source file:** `data/crypto_raw.csv`  
**Period:** 2013-04-29 → 2021-07-06  
**Currencies:** 23 (AAVE, ADA, ATOM, BNB, BTC, CRO, DOGE, DOT, EOS, ETH, LINK, LTC, MIOTA, SOL, TRX, UNI, USDC, USDT, WBTC, XEM, XLM, XMR, XRP)  
**Total records:** 37,082

---

## 1. Data Quality

- **No null values** in any numeric column — the `load_data.py` pipeline already removed them.
- **No duplicates** (Symbol, Date).
- Very uneven temporal coverage: BTC has 2,991 records (since 2013) while AAVE has only 275 (since 2020). For supervised models, it is recommended to use **BTC, ETH, LTC, XRP, and XMR** which have the greatest historical coverage.
- Available columns are: `SNo`, `Name`, `Symbol`, `Date`, `High`, `Low`, `Open`, `Close`, `Volume`, `Marketcap`.

---

## 2. Descriptive Statistics of Daily Returns

| Currency | Std (volatility) | Skewness | Kurtosis | Observation |
|----------|:----------------:|:--------:|:--------:|-------------|
| XEM      | high             | 16.04    | 257.4    | Single extreme event |
| DOGE     | high             | 37.12    | 1377.0   | Extreme crashes 2021 |
| BTC      | low-medium       | 0.24     | 10.27    | Most stable in the group |
| ETH      | low-medium       | 0.86     | 7.85     | Second most stable |
| USDT     | very low         | 11.72    | 1042.7   | Stablecoin |
| USDC     | very low         | 0.58     | 19.4     | Stablecoin |

**Conclusion:** No cryptocurrency follows a normal distribution. All exhibit **leptokurtosis** (heavy tails). QQ-plots confirm that extreme returns are much more frequent than predicted by the normal distribution.

---

## 3. Time Series and Prices

- BTC reached its all-time high (in this dataset) of ~$58,000 in Feb-2021.
- ETH and BNB showed exponential growth in 2020-2021, surpassing BTC in terms of percentage return since their first available data.
- **Volume** of BTC dominates the market; historical peaks were recorded in 2020-2021.
- Normalized price (base=100): BNB was the asset with the highest relative performance in the period, followed by ETH.

---

## 4. Correlations between Currencies

**Most correlated pairs (daily returns):**
1. WBTC–BTC: 0.93 — practically the same asset (Wrapped Bitcoin)
2. WBTC–ETH: 0.74
3. WBTC–LTC: 0.71
4. UNI–AAVE: 0.69 — both DeFi tokens
5. ETH–EOS: 0.68

**Least correlated pairs:**
- USDT and USDC with any non-stable cryptocurrency: correlations close to zero or negative.
- XEM and DOGE show low correlations with the rest (idiosyncratic events dominant).

**Implication for models:** WBTC and BTC are redundant as features — use only one. Stablecoins contribute little as price predictors but can be useful for detecting market context.

---

## 5. Rolling Volatility

- **Volatility peaks** identified:
  - 2017-2018: bull run and subsequent correction
  - Mar-2020: COVID-19 crash (BTC vol_30d reached ~8%)
  - Jan-Feb 2021: bullish rally with increased volatility
- 90-day volatility correctly smooths short-term signals.
- DOGE and small altcoins have structurally higher volatility than BTC and ETH.

---

## 6. Outlier Detection

| Currency | Z-Score Outliers | % | IQR Outliers | % |
|----------|:----------------:|:-:|:------------:|:-:|
| DOGE     | ~15              |~6%| ~60         |~23%|
| XEM      | ~10              |~4%| ~50         |~20%|
| BTC      | ~8               |~0.3%|~100      |~4% |
| ETH      | ~6               |~0.3%|~90       |~4% |

- Z-Score detects the most extreme events; IQR is more aggressive and captures more "unusual" days.
- BTC outliers coincide with identifiable historical events: COVID crash (Mar-2020), 2017 rally, 2018 crash.
- **Engineering decision:** Do not remove outliers from the dataset; add them as binary features (`outlier_z`, `outlier_iqr`) so the model learns the context of extreme market conditions.

---

## 7. Returns Distribution

- All returns show **positive skewness** (except ATOM with -0.09 and WBTC with -0.01).
- Right tails (extreme positive returns) are heavier than left ones in most cases.
- QQ-plots show clear deviation from normality in the tails.
- **Implication for models:** Models based on normality assumption (e.g., pure OLS) will be miscalibrated. Prefer robust models (Random Forest, XGBoost) or transform returns with `np.sign(r) * np.log1p(|r|)`.

---

## 8. Seasonality

- **By month:** In BTC, January, April, and November tend to have positive average returns. February and September tend to be negative. The pattern is not consistent across currencies.
- **By day of week:** Crypto assets operate 7 days. There is no clear "Monday effect" as in stock markets, but some assets show slightly higher returns on weekends.
- **Conclusion:** Seasonality is weak. Using month and day of week variables as dummy features can provide marginal signal but will not be the main driver of the models.

---

## Decisions for the Supervised Phase

1. **Priority currencies:** BTC and ETH (greater history, less noise from extreme outliers). Model separately or include Symbol as categorical feature.
2. **Features to create:** SMA 7/14/30, EMA 14, RSI 14, Bollinger Bands, MACD, vol_30d, vol_90d, lag-1 return, lag-2, lag-5.
3. **Target:** `Close(t+1) > Close(t)` → binary classification with `target=1`.
4. **Temporal split:** 80/10/10 without shuffle — data has temporal structure and cannot be mixed.
5. **WBTC vs BTC:** Do not include both in the same model — they are collinear (r=0.93).
6. **Outliers:** Do not remove; include as binary features.
7. **Stablecoins (USDT, USDC):** Exclude as price prediction targets; they can be useful as context.