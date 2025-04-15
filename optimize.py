import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import generate_features

# === Load training data ===
train_df = pd.read_csv("data/train.csv")
transactions = pd.read_csv("data/transactions.csv")
transfers = pd.read_csv("data/token_transfers.csv")
swaps = pd.read_csv("data/dex_swaps.csv")

train_addresses = train_df["ADDRESS"].tolist()
full_df = generate_features(train_addresses, [], transactions, transfers, swaps)

X = full_df.loc[train_addresses].fillna(0)
y = train_df.set_index("ADDRESS").loc[X.index]["LABEL"]

# === Optuna objective ===
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2),
        'max_depth': trial.suggest_int("max_depth", 3, 12),
        'num_leaves': trial.suggest_int("num_leaves", 15, 255),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 0, 5.0),
        'reg_lambda': trial.suggest_float("reg_lambda", 0, 5.0),
        'verbose': -1
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, y_train)
        dval = lgb.Dataset(X_val, y_val)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        preds = model.predict(X_val)
        score = roc_auc_score(y_val, preds)
        scores.append(score)

    return sum(scores) / len(scores)

# === Start tuning ===
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("✅ Best parameters:", study.best_params)
