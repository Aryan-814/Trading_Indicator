# References and Citations

The following papers, libraries, and frameworks are cited and utilized in this project per spec section 21:

## Academic Papers

1. **Mesfin (2026)**. *Sequential Structure in Intraday Futures Data: LSTM vs Gradient Boosting on MNQ*. arXiv:2605.17724.
   - Highlights: Intraday 5-minute directional prediction accuracy sits near 50.00% to 50.89%; emphasizes selective coverage and calibration.

2. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley.
   - Concepts used: Purged walk-forward cross-validation, embargo, meta-labeling for trade sizing/confidence.

3. **Bailey, D. H. & López de Prado, M. (2014)**. *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*. Journal of Portfolio Management, 40(5), 94-107.
   - Concepts used: Deflated Sharpe ratio adjusting for trial count in hyperparameter search.

4. **Joubert, J. F. (2022)**. *Meta-Labeling: Theory and Framework*. Journal of Financial Data Science, 4(3).
   - Concepts used: Secondary model predicting binary correctness of primary directional signal.

5. **Gibbs, I. & Candès, E. (2021)**. *Adaptive Conformal Inference Under Distribution Shift*. NeurIPS. arXiv:2106.00170.
   - Concepts used: Online adaptive conformal prediction interval calibration under market drift.

6. **Ke, G. et al. (2017)**. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
   - Concepts used: Tree-based gradient boosting with histogram binning and native feature contribution calculation (`pred_contrib=True`).

7. **Zadrozny, B. & Elkan, C. (2002)**. *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*. KDD.
   - Concepts used: Non-parametric isotonic regression calibration mapping raw probabilities to empirical frequencies.

## Libraries and Tools

- **LightGBM**: Primary classifier, meta-classifier, and quantile regressors.
- **scikit-learn**: Isotonic regression calibrator and validation metrics (Brier score, AUC, log-loss).
- **Alpaca Market Data API (`alpaca-py`)**: IEX real-time and historical 5-minute bar feeds.
- **Pandas / NumPy / PyArrow**: Feature engineering, data alignment, Parquet storage.
- **PyYAML**: Single-source-of-truth configuration management.

