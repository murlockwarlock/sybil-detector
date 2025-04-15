import lightgbm as lgb
import matplotlib.pyplot as plt

def plot_feature_importance(model_path="models/model_fold0.txt", max_features=30):
    model = lgb.Booster(model_file=model_path)
    ax = lgb.plot_importance(model, max_num_features=max_features, importance_type='gain', figsize=(10, 6))
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("✅ Saved: feature_importance.png")
