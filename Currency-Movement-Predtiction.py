#%% =========================
# 0. IMPORTS
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore")

#%% =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Global Trade War\archive\currency_impact.csv")
df = df.sort_values(["currency", "country", "date"]).reset_index(drop=True)

#%% =========================
# 2. CLEANING
# =========================
df = df.dropna(subset=["pct_change_1d"])

#%% =========================
# 3. FILTERING (NOISE REDUCTION)
# =========================

df["vol_7d"] = df["rolling_7d_vol"]
df["vol_30d"] = df["rolling_30d_avg"]


# filter noisy low-volatility days (IMPORTANT)
df = df[df["vol_7d"] > df["vol_7d"].quantile(0.4)]

#%% =========================
# 4. TARGET (CLASSIFICATION)
# =========================
df["target"] = (
    df.groupby(["currency", "country"])["pct_change_1d"].shift(-1) > 0
).astype(int)

df = df.dropna(subset=["target"])

#%% =========================
# 5. FEATURE ENGINEERING
# =========================

# LAGS (better expanded set)
for lag in [1, 2, 3, 5, 7, 10, 14]:
    df[f"lag_{lag}"] = df.groupby(["currency", "country"])["pct_change_1d"].shift(lag)

# MOMENTUM FEATURES (VERY IMPORTANT)
df["momentum_7_1"] = df["pct_change_7d"] - df["pct_change_1d"]
df["momentum_30_7"] = df["pct_change_30d"] - df["pct_change_7d"]

df["trend_strength"] = df["pct_change_7d"] / (df["vol_7d"] + 1e-8)
df["vol_ratio"] = df["vol_7d"] / (df["vol_30d"] + 1e-8)

# CLEAN NA
df = df.dropna()

#%% =========================
# 6. FEATURES
# =========================
features = [
    "pct_change_1d",
    "pct_change_7d",
    "pct_change_30d",
    "rolling_7d_vol",
    "rolling_30d_avg",

    "lag_1",
    "lag_2",
    "lag_3",
    "lag_5",
    "lag_7",
    "lag_10",
    "lag_14",

    "momentum_7_1",
    "momentum_30_7",
    "trend_strength",
    "vol_ratio",

   

    "currency",
    "country"
]

target = "target"
cat_features = ["currency", "country"]
#%%new deleting some records


# STEP 1: compute valid countries
country_counts = df["country"].value_counts()
valid_countries = country_counts[country_counts >= 2000].index

# STEP 2: filter AND overwrite safely
df = df.loc[df["country"].isin(valid_countries)].copy()

# STEP 3: verify immediately
print(df["country"].value_counts())

#%% =========================
# 7. TRAIN / TEST SPLIT (TIME SAFE)
# =========================
train_list = []
test_list = []

for (currency, country), group in df.groupby(["currency", "country"]):
    group = group.sort_values("date")

    split = int(len(group) * 0.8)

    train_list.append(group.iloc[:split])
    test_list.append(group.iloc[split:])

train_df = pd.concat(train_list)
test_df = pd.concat(test_list)

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

#%% =========================
# 8. MODEL TRAINING
# =========================
model = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=200
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    cat_features=cat_features,
    early_stopping_rounds=80
)

#%% =========================
# 9. PREDICTIONS (CONFIDENCE FILTER)
# =========================
proba = model.predict_proba(X_test)[:, 1]

# confidence-based filtering (VERY IMPORTANT)
threshold_high = 0.51
threshold_low = 0.49

mask = (proba > threshold_high) | (proba < threshold_low)

y_pred = (proba > 0.5).astype(int)

# evaluate only confident predictions
y_test_filtered = y_test[mask]
y_pred_filtered = y_pred[mask]

#%% =========================
# 10. EVALUATION
# =========================
acc = accuracy_score(y_test_filtered, y_pred_filtered)

print("CONFIDENCE-FILTERED ACCURACY:", round(acc, 4))
print("\nReport:\n")
print(classification_report(y_test_filtered, y_pred_filtered))



#%%
# #%% =========================
# 11. FEATURE IMPORTANCE
# =========================

importances = model.get_feature_importance()

feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feat_imp,
    x="importance",
    y="feature"
)

plt.title("CatBoost Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.grid(True)
plt.show()
#%%
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test_filtered, y_pred_filtered)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["DOWN", "UP"],
    yticklabels=["DOWN", "UP"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()
#%% =========================
# 12. SAVE MODEL
# =========================
model.save_model(
    r"C:\Users\Lenovo\Desktop\Global Trade War\currency_improved_model.cbm"
)
# %%
