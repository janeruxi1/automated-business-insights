| Model | MAE (orders/day) | RMSE | MAPE | sMAPE | MASE | Folds beating baseline (of 7) | MAE vs. baseline |
|---|---|---|---|---|---|---|---|
| Prophet (weekly + BR holidays) | 38.9 | 46.3 | 21.0% | 17.8% | 1.19 | 6 | -30.5% |
| Naive (last day) | 51.1 | 60.1 | 26.9% | 23.2% | 1.56 | 3 | -8.6% |
| Seasonal naive (last week) | 55.9 | 67.7 | 26.7% | 27.8% | 1.71 | — | — |

Excluding fold 1 (whose baseline copies the post-New Year lull), Prophet's MAE is 41.1 vs. 49.8 for seasonal naive (-17.5%).
