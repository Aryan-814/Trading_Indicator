# References and Citations

The following papers, preprints, libraries, and frameworks are cited and utilized in the design, modeling, calibration, and evaluation of this trading indicator system:

---

## 1. Primary Academic Sources & Research Preprints

1. **Mesfin, T. (2026)**. *Sequential Structure in Intraday Futures Data: LSTM vs Gradient Boosting on MNQ*. arXiv:2605.17724.  
   - **Type**: Research Preprint (arXiv). *Note: Preprints may not have completed peer review.*
   - **Relevance**: Demonstrates that state-of-the-art architectures on 5-minute intraday financial bars achieve approximately 50.00% to 50.89% directional accuracy, establishing the theoretical necessity of selective prediction, abstention, and cost-aware evaluation.

2. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. John Wiley & Sons.  
   - **Type**: Peer-Reviewed Book / Monograph.
   - **Relevance**: Informs the triple-barrier labeling methodology, the purged and embargoed walk-forward cross-validation splitter (preventing information leakage across autocorrelated bars), and the meta-labeling formulation.

3. **Bailey, D. H. & López de Prado, M. (2014)**. *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*. Journal of Portfolio Management, 40(5), 94–107.  
   - **Type**: Peer-Reviewed Journal Article.
   - **Relevance**: Motivates strict tracking of total trial counts across hyperparameter exploration to account for multiple hypothesis testing and selection bias.

4. **Joubert, J. F. (2022)**. *Meta-Labeling: Theory and Framework*. Journal of Financial Data Science, 4(3).  
   - **Type**: Peer-Reviewed Journal Article.
   - **Relevance**: Establishes mathematical foundations for decoupling directional classification (primary model) from trade sizing / abstention filtering (secondary meta-model).

5. **Gibbs, I. & Candès, E. (2021)**. *Adaptive Conformal Inference Under Distribution Shift*. Advances in Neural Information Processing Systems (NeurIPS 34), 1660–1672.  
   - **Type**: Peer-Reviewed Conference Proceedings.
   - **Relevance**: Informs the study of conformal prediction coverage under non-stationary market regimes, justifying why offline calibration is preferred by default to avoid tracking noisy intraday regime shifts.

6. **Temporal Conformal Prediction (2025)**. arXiv:2507.05470.  
   - **Type**: Research Preprint (arXiv). *Note: Preprints may not have completed peer review.*
   - **Relevance**: Analysis of conformal prediction intervals for time-series forecasting.

7. **FinAbstain (2026)**. arXiv:2607.24875.  
   - **Type**: Research Preprint (arXiv). *Note: Preprints may not have completed peer review.*
   - **Relevance**: Informs the risk-coverage performance evaluation (selective accuracy as a function of answered candle coverage).

8. **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. & Liu, T. Y. (2017)**. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS 30), 3146–3154.  
   - **Type**: Peer-Reviewed Conference Proceedings.
   - **Relevance**: Core gradient-boosted tree framework used for primary directional prediction, meta-probability estimation, quantile regressions ($q_{10}, q_{50}, q_{90}$), and tree feature contributions (`pred_contrib=True`).

9. **Fischer, T. & Krauss, C. (2018)**. *Deep learning with long short-term memory networks for financial market predictions*. European Journal of Operational Research, 270(2), 654–669.  
   - **Type**: Peer-Reviewed Journal Article.
   - **Relevance**: Demonstrates that statistical edge in high-frequency directional signals declines substantially once transaction costs and slippage are incorporated.

10. **Zadrozny, B. & Elkan, C. (2002)**. *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*. Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 694–699.  
    - **Type**: Peer-Reviewed Conference Proceedings.
    - **Relevance**: Non-parametric isotonic regression algorithm used for calibrating raw meta probabilities into empirical confidence scores.

11. **Wilder, J. W. (1978)**. *New Concepts in Technical Trading Systems*. Trend Research.  
    - **Type**: Technical Monograph.
    - **Relevance**: Exponential smoothing formulation of the Relative Strength Index (`rsi_14`) and standard RSI baseline comparison rule.

12. **Wilson, E. B. (1927)**. *Probable inference, the law of succession, and statistical inference*. Journal of the American Statistical Association, 22(158), 209–212.  
    - **Type**: Peer-Reviewed Journal Article.
    - **Relevance**: Binomial proportion confidence intervals using effective sample size $n_{\text{eff}} = n / h$ for overlapping prediction horizons.

---

## 2. Core Libraries & Documentation

- **TradingView Lightweight Charts (v4.2.0)**: Pinned standalone JavaScript charting engine (`lightweight-charts.standalone.production.js`).
- **Alpaca Market Data API (`alpaca-py`)**: Real-time IEX WebSocket and historical REST bar client.
- **FastAPI / Uvicorn**: Async Python web framework, REST endpoints, and WebSocket broadcasting.
- **SQLite 3 (WAL mode)**: Persistence engine with database-level immutability triggers.
- **Scikit-Learn**: Calibrator regression and validation performance metrics.
- **Pandas / NumPy / PyArrow**: Feature engineering, time-series alignment, and columnar storage.
