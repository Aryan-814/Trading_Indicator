# Phase 9 Report: Documentation, Final Verification, & Model Freeze

**Date:** November 2026  
**Status:** Complete & Frozen  
**Exit Gate Status:** All documentation complete; all numbers reproducible from repo; model bundle frozen and hashed; final acceptance checklist verified.

---

## 1. Documentation Deliverables

1. **User Guide ([`docs/user_guide.md`](file:///home/aryan/Sem_5/ML_project/docs/user_guide.md))**:
   - Outlines signal mechanics, dynamic horizon ({horizon_minutes} minutes from bundle), one-second readability, calibrated confidence, regime warnings, and honest expectations.
2. **References & Citations ([`docs/references.md`](file:///home/aryan/Sem_5/ML_project/docs/references.md))**:
   - Complete academic citations (López de Prado 2018, Joubert 2022, Ke et al. 2017, Zadrozny & Elkan 2002, Fischer & Krauss 2018), clearly identifying preprints as preprints (Mesfin 2026, FinAbstain 2026, Temporal Conformal Prediction 2025).
3. **Demo Script ([`docs/demo_script.md`](file:///home/aryan/Sem_5/ML_project/docs/demo_script.md))**:
   - 5–7 minute evaluator walkthrough script with prepared technical responses for questions regarding accuracy near 50%, confidence calibration, abstention philosophy, leakage prevention, and immutability guarantees.
4. **README ([`README.md`](file:///home/aryan/Sem_5/ML_project/README.md))**:
   - Complete end-to-end instructions for data cleaning, model training, running tests, launching the live backend, and full-day replay.

---

## 2. Model Freeze & Bundle Verification

The production LightGBM model bundle is frozen at `models/v1/`:
- **Model Version**: `v1`
- **Primary Model**: `models/v1/primary.txt` (SHA-256: `18f5043bf7e992147d3cfa3200a7faad147f7d142d547514be80687d46816fa8`)
- **Meta Model**: `models/v1/meta.txt` (SHA-256: `3b2a7596ff506ba567c9c0b5e408ec2e76550bf2ceb873f73df0169fc03f9dc2`)
- **Calibrator**: `models/v1/calibrator.joblib` (SHA-256: `24b531405c907b226f987f620e2ef64d06a0904d9c720516625890bb86256f6b`)
- **Key Hyperparameters**:
  - `horizon_bars`: 12 (60 minutes forward)
  - `kdz`: 0.30
  - `tau_m`: 0.5218
  - `skip_first_bars`: 3
  - `target_coverage`: 0.20
  - `feature_count`: 24 causal features
- **Holdout Evaluation**: Evaluated strictly once under Phase 2 lock; audit recorded in `reports/holdout_log.json`.

