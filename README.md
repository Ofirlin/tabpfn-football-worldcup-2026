# World Cup 2026 — TabPFN ensemble entry (Ofirlin)

Entry for Prior Labs' [World Cup Game Outcome Prediction competition](https://ux.priorlabs.ai/football).
Predicts international match outcomes (home win / draw / away win) and is scored on
multi-class log-loss against the 90-minute result.

## Model: LASSO-select top-K + TabPFN ensemble

A reproducible pipeline built on **TabPFN** (run locally on GPU):

1. **Features** — the leakage-free engineered features from the starter template
   (ELO with goal-difference & tournament-importance K-factor, recent form, goal stats,
   streaks, rest, head-to-head, neutral venue), built in a single chronological pass
   (`predict_local.py`).
2. **Feature expansion** — tabprep-style row-wise arithmetic expansion to 389 features
   (squares, all pairwise products, curated ratios) in `features_plus.py`.
3. **LASSO selection + prediction** — an L1-multinomial logistic regression is fit on the
   expanded features; it acts as both a predictor and a selector of the **top-K** features
   by aggregated |coefficient| (K=15, tuned by walk-forward validation log-loss).
4. **TabPFN-2.5** is trained on those top-K features.
5. **Ensemble** — final probabilities are the mean of the LASSO and TabPFN predictions.

## Reproduce

```bash
pip install -r requirements.txt
python predict_ensemble.py        # writes predictions_ensemble_<date>.csv
```

`predict_ensemble.py` downloads the latest [martj42/international_results](https://github.com/martj42/international_results)
dataset, builds features, trains on the most recent 10 000 matches, and predicts all
upcoming fixtures.

## Backtesting

Walk-forward (retrain each month, predict the held-out month) over 2016–2026:

| Model | Held-out accuracy | Log-loss |
|---|---|---|
| TabPFN (base 26 features) | ~60% | ~0.86 |
| LASSO top-K + TabPFN ensemble | ~60% | ~0.86 |

- `backtest_walkforward.py` — TabPFN walk-forward baseline.
- `backtest_lasso_tabpfn.py` — the LASSO-select + TabPFN ensemble, with K tuned on a
  validation span and reported on a held-out test span.

## Files

| File | Purpose |
|---|---|
| `predict_ensemble.py` | Generate predictions for upcoming fixtures (submission model) |
| `predict_local.py` | Data loading + leakage-free feature engineering (local TabPFN) |
| `features_plus.py` | tabprep-style arithmetic feature expansion |
| `backtest_lasso_tabpfn.py` | Walk-forward backtest of the LASSO+TabPFN ensemble |
| `backtest_walkforward.py` | Walk-forward TabPFN baseline |
| `predictions_ensemble_20260628.csv` | Predictions for the current knockout fixtures |
