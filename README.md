# 🛡️ Sybil Detection - Human Passport x Octant Submission

This repository contains the full solution by [@xlisttop1mmr](https://x.com/xlisttop1mmr) for the Sybil Detection challenge hosted by Holonym, Octant, and Ethereum Foundation.

---

## 🎯 Objective

Detect Sybil wallets among 10,000+ test addresses using historical on-chain data (Ethereum & Base).  
The model outputs a probability (0 = non-Sybil, 1 = Sybil) for each address.

📁 data/ folder contains private datasets used during training. Not included in the repo.



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
```
Git clone:
```bash
git clone https://github.com/murlockwarlock/sybil-detector.git
cd sybil-detector
```


🛠 Install dependencies:
```bash
pip install -r requirements.txt
```
🧠 Train model:
```bash
python train.py
```
🔍 Generate predictions:
```bash
python predict.py
```

🧪 Visualize feature importances:
```bash
python feature_importance.py
```

🧩 Feature Categories
TX Features: count, lifespan, ETH volume, chain separation

Token Features: number of unique tokens, direction ratio

Swap Features: volume, diversity, same-token spam detection

Graph Features: neighbors, edge count

Multi-Chain Flag: is wallet active on both Ethereum & Base?

📤 Output Format
```bash
ADDRESS,PRED
0xabc...,0.921
0xdef...,0.018
...
```

🙌 Special Thanks

Big thanks to Holonym, Octant, and Ethereum Foundation for supporting decentralized Sybil defense.

Created by: @xlisttop1mmr / @Xlistyara


