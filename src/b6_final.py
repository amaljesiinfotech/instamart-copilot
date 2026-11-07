import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from lapse_features import get_split, make_preprocessor

SELECTED = "rf"   

df, X_train, X_test, y_train, y_test = get_split()
best = json.load(open("outputs/best_params.json"))


def make_lr():
    return Pipeline([("pre", make_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000, C=best["C"]))])


def make_rf():
    return Pipeline([("pre", make_preprocessor()),
                     ("clf", RandomForestClassifier(
                         n_estimators=100, random_state=42,
                         max_depth=best["max_depth"],
                         min_samples_leaf=best["min_samples_leaf"]))])


def scores(model):
    p = model.predict(X_test)
    return (round(precision_score(y_test, p, zero_division=0), 3),
            round(recall_score(y_test, p, zero_division=0), 3),
            round(f1_score(y_test, p, zero_division=0), 3))


lr = make_lr().fit(X_train, y_train)
rf = make_rf().fit(X_train, y_train)

# ---------- Task 7: oversample TRAINING rows only ----------
n_not = int((y_train == 0).sum())
n_lap = int((y_train == 1).sum())
lap_idx = y_train[y_train == 1].index.to_numpy()
extra = list(resample(lap_idx, replace=True, n_samples=n_not - n_lap, random_state=42))
X_bal = pd.concat([X_train, X_train.loc[extra]])
y_bal = pd.concat([y_train, y_train.loc[extra]])
print("balanced training counts:", y_bal.value_counts().to_dict(), "(expected 570 each)")

lr_bal = make_lr().fit(X_bal, y_bal)
print("\n=== Class balance experiment (test set, lapsed class) ===")
print("                  precision, recall, F1")
print("LR un-balanced :", scores(lr))
print("LR oversampled :", scores(lr_bal))
print("Expected: recall about 0.64 -> 0.85, precision stays near 0.6")

# ---------- Task 8a: interpretation ----------
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
print("\n=== Top 5 random forest features ===")
print(imp.sort_values(ascending=False).head(5).round(3))

coef = pd.Series(lr.named_steps["clf"].coef_[0],
                 index=lr.named_steps["pre"].get_feature_names_out())
print("\n=== Logistic regression coefficients (3 features) ===")
for f in ["num__recency", "num__frequency", "num__cancel_rate"]:
    print(f, round(coef[f], 3), "->", "raises" if coef[f] > 0 else "lowers", "lapse risk")

# ---------- Task 8b: save, reload, compare ----------
model = rf if SELECTED == "rf" else lr
print("\nSelected model:", "Random Forest" if SELECTED == "rf" else "Logistic Regression",
      "| test (precision, recall, F1):", scores(model))
joblib.dump(model, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(model.predict(X_test), reloaded.predict(X_test))
print("Reloaded model reproduces test predictions exactly:", same)
assert same

# ---------- Task 8c: retention list (illustrative) ----------
seg = pd.read_csv("outputs/customer_segments.csv")
premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
test_info = pd.DataFrame({
    "customer_id": df.loc[X_test.index, "customer_id"].values,
    "lapse_probability": model.predict_proba(X_test)[:, 1].round(4),
    "actually_lapsed": y_test.values})
prem = test_info[test_info["customer_id"].isin(premium_ids)]
top = prem.sort_values("lapse_probability", ascending=False).head(10)
top.to_csv("outputs/retention_list.csv", index=False)
print("\nPremium customers in test set:", len(prem))
print("Retention list rows:", len(top), "| truly lapsed:", int(top["actually_lapsed"].sum()))
print(top.to_string(index=False))