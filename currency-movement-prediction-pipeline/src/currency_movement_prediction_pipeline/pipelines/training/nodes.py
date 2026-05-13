import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


# =========================
# SPLIT DATA
# =========================
def split_data(df, params):

    # sort + reset index (important for time series stability)
    df = df.sort_values("date").reset_index(drop=True)

    target = params["target_column"]

    split = int(len(df) * params["train_fraction"])

    train = df.iloc[:split]
    test = df.iloc[split:]

    # avoid leakage: exclude target + date explicitly
    features = [
        c for c in df.columns
        if c not in [target, "date"]
    ]

    return (
        train[features],
        test[features],
        train[target],
        test[target],
    )


# =========================
# TRAIN MODEL
# =========================
def train_model(x_train, y_train, params):

    model = CatBoostClassifier(
        **params["model_params"]["catboost"]
    )

    # safe categorical feature handling
    cat_features = [
        c for c in ["currency", "country"]
        if c in x_train.columns
    ]

    model.fit(
        x_train,
        y_train,
        cat_features=cat_features if cat_features else None,
        verbose=False
    )

    return model


# =========================
# PREDICT
# =========================
def predict(model, x_test):

    proba = model.predict_proba(x_test)[:, 1]

    return pd.DataFrame(
        {
            "prediction": (proba > 0.5).astype(int),
            "probability": proba,
        },
        index=x_test.index  # keeps alignment with y_test
    )


# =========================
# METRICS
# =========================
def compute_metrics(y_true, predictions):

    acc = accuracy_score(
        y_true,
        predictions["prediction"]
    )

    return {
        "accuracy": float(acc)
    }