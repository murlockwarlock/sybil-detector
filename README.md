# 🛡️ Sybil Detection - Human Passport x Octant Submission

This repository contains the full solution by [@xlisttop1mmr](https://x.com/xlisttop1mmr) for the Sybil Detection challenge hosted by Holonym, Octant, and Ethereum Foundation.

---

## 🎯 Objective

Detect Sybil wallets among 10,000+ test addresses using historical on-chain data (Ethereum & Base).  
The model outputs a probability (0 = non-Sybil, 1 = Sybil) for each address.

---

## 🧠 Model Highlights

- 📊 LightGBM classifier with full-feature engineering pipeline
- 🔁 Stratified 5-fold cross-validation
- ⚙️ Optuna for automatic hyperparameter tuning
- 🕸️ Graph-based features (number of neighbors, unique counterparties)
- 🌐 Chain-aware behavior (Ethereum vs Base)
- 🧪 Final output: `submission.csv` for leaderboard evaluation

---

## 📂 Structure

```bash
sybil-detector/
├── config.yaml              # Model config
├── features.py              # Feature engineering
├── graph_features.py        # Wallet graph features
├── train.py                 # Training pipeline
├── predict.py               # Inference script
├── optimize.py              # Optuna optimization
├── feature_importance.py    # Feature visualizer
├── submission.csv           # Final predictions
├── requirements.txt
├── README.md
├── models/                  # Saved model files
└── data/
    ├── train.csv
    ├── test_addresses.csv
    ├── transactions.csv
    ├── token_transfers.csv
    └── dex_swaps.csv
