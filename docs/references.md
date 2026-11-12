# References and Citations

The following papers, libraries, and frameworks are cited and utilized in this project:

## Academic Papers

1. **Mesfin (2026)**. *Sequential Structure in Intraday Futures Data: LSTM vs Gradient Boosting on MNQ*. arXiv:2605.17724.
   - Relevance: Intraday 5-minute directional prediction accuracy sits near 50.00% to 50.89%; motivates selective coverage and calibration.

2. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley.
   - Concepts used: Purged walk-forward cross-validation and meta-labeling for trade selection/confidence.

3. **Bailey, D. H. & López de Prado, M. (2014)**. *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*. Journal of Portfolio Management, 40(5), 94–107.
   - Relevance: Theoretical motivation for multiple-testing correction (implemented via Bonferroni-adjusted p-values across total trial count rather than DSR).

4. **Joubert, J. F. (2022)**. *Meta-Labeling: Theory and Framework*. Journal of Financial Data Science, 4(3).
   - Concepts used: Secondary model predicting binary correctness of the primary directional classifier.

5. **Ke, G. et al. (2017)**. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
   - Concepts used: Primary, meta, and quantile gradient boosted trees plus native feature contributions (`pred_contrib=True`).

6. **Zadrozny, B. & Elkan, C. (2002)**. *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*. KDD.
   - Concepts used: Non-parametric isotonic regression calibration mapping raw meta probabilities to empirical accuracy.

7. **Wilder, J. W. (1978)**. *New Concepts in Technical Trading Systems*. Trend Research.
   - Concepts used: Wilder's exponential smoothing Relative Strength Index (`rsi_14`) and RSI baseline rule.

8. **Wilson, E. B. (1927)**. "Probable inference, the law of succession, and statistical inference." *Journal of the American Statistical Association*, 22(158), 209–212.
   - Concepts used: Wilson score confidence intervals using effective sample size `n / h`.

## Libraries and Tools

- **LightGBM**: Primary classifier, meta-classifier, and quantile regressors (`.txt` native format).
- **scikit-learn**: Isotonic regression calibrator (`IsotonicRegression`), logistic baseline (`LogisticRegression`), and evaluation metrics (`roc_auc_score`, `brier_score_loss`).
- **Alpaca Market Data API (`alpaca-py`)**: IEX historical 5-minute bar feeds.
- **Pandas / NumPy / PyArrow**: Feature engineering, data alignment, Parquet storage.
- **PyYAML**: Single-source-of-truth configuration management.
