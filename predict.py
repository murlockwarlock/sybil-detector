import pandas as pd
import lightgbm as lgb
from features import generate_features

# === Load test addresses and input data ===
test_df = pd.read_csv("data/test_addresses.csv")
transactions = pd.read_csv("data/transactions.csv")
transfers = pd.read_csv("data/token_transfers.csv")
swaps = pd.read_csv("data/dex_swaps.csv")

# === Generate features ===
test_addresses = test_df["ADDRESS"].tolist()
features = generate_features([], test_addresses, transactions, transfers, swaps)
X_test = features.loc[test_addresses].fillna(0)

# === Load models ===
models = []
for i in range(5):
    model = lgb.Booster(model_file=f"models/model_fold{i}.txt")
    models.append(model)

# === Inference ===
preds = sum(model.predict(X_test) for model in models) / len(models)

# === Save submission ===
submission = pd.DataFrame({
    "ADDRESS": test_addresses,
    "PRED": preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Saved: submission.csv")
