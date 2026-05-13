import pandas as pd


def predict_per_country(model, df):

    results = []

    for country, group in df.groupby(["country"]):

        X = group.drop(columns=["target"], errors="ignore")

        proba = model.predict_proba(X)[:, 1]

        group = group.copy()

        group["prediction"] = (proba > 0.5).astype(int)
        group["probability"] = proba

        results.append(group)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()