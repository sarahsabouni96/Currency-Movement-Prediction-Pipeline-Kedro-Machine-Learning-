import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier


# =========================================================
# LOAD DATA
# =========================================================
def load_data(df: pd.DataFrame):

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    return (
        df.sort_values(["country", "currency", "date"])
        .reset_index(drop=True)
    )


# =========================================================
# CLEAN DATA
# =========================================================
def clean_data(df: pd.DataFrame):

    return df.dropna(subset=["pct_change_1d"])


# =========================================================
# FILTER DATA
# =========================================================
def filter_data(df: pd.DataFrame):

    df = df.copy()

    df["vol_7d"] = df["rolling_7d_vol"]
    df["vol_30d"] = df["rolling_30d_avg"]

    threshold = df["vol_7d"].quantile(0.4)

    return df[df["vol_7d"] > threshold]


# =========================================================
# RENAME COLUMNS
# =========================================================
def rename_columns(df, renaming_dict):

    return df.rename(columns=renaming_dict)


# =========================================================
# CREATE TARGET
# =========================================================
def create_target(df):

    df = df.copy()

    df["target"] = (
        df.groupby(["currency", "country"])["pct1"]
        .shift(-1) > 0
    ).astype(int)

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def get_features(df, lag_params):

    df = df.copy()

    df = df.sort_values(
        ["country", "currency", "date"]
    ).reset_index(drop=True)

    lags = lag_params["lags"]

    feature_cols = [
        "pct1",
        "pct7",
        "pct30",
    ]

    for feature in feature_cols:

        for lag in lags:

            df[f"{feature}_lag_{lag}"] = (
                df.groupby(["country", "currency"])[feature]
                .shift(lag)
            )

    df["mom7"] = df["pct7"] - df["pct1"]
    df["mom30"] = df["pct30"] - df["pct7"]

    df = df.dropna().reset_index(drop=True)

    timestamps = df["date"]

    return df, timestamps


# =========================================================
# TRAIN TEST SPLIT
# =========================================================
def split_data(df, params):

    target_name = params["target_params"]["new_target_name"]

    df = df.sort_values("date")

    split_index = int(len(df) * params["train_fraction"])

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    drop_cols = [
        target_name,
        "date",
    ]

    features = [
        c for c in df.columns
        if c not in drop_cols
    ]

    return (
        train_df[features],
        test_df[features],
        train_df[target_name],
        test_df[target_name],
    )


# =========================================================
# TRAIN MODEL
# =========================================================
def train_model(x_train, y_train, params):

    model_type = params["model_type"]

    model_params = params["model_params"][model_type]

    model = CatBoostClassifier(**model_params)

    model.fit(
        x_train,
        y_train,
        cat_features=["currency", "country"],
        verbose=200,
    )

    return model


# =========================================================
# PREDICT
# =========================================================
def predict(model, x_test):

    proba = model.predict_proba(x_test)[:, 1]

    predictions = (proba > 0.5).astype(int)

    return pd.DataFrame(
        {
            "prediction": predictions,
            "probability": proba,
        }
    )


# =========================================================
# SAVE MODEL
# =========================================================
def save_model(model, model_storage):

    model_dir = Path(model_storage["path"])

    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_storage["name"]

    model.save_model(
        str(model_dir / f"{model_name}.cbm")
    )