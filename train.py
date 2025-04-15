for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=config["num_boost_round"],
        valid_sets=[dval],
        early_stopping_rounds=config["early_stopping_rounds"],
        verbose_eval=False,
    )

    # Save model per fold
    os.makedirs("models", exist_ok=True)
    model.save_model(f"models/model_fold{fold}.txt")

    preds = model.predict(X_val)
    score = roc_auc_score(y_val, preds)
    auc_scores.append(score)
    print(f"Fold {fold} AUC: {score:.4f}")
    models.append(model)
