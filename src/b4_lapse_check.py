import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from lapse_features import NUM, CAT, get_split, make_preprocessor

df, X_train, X_test, y_train, y_test = get_split()

print("cohort:", len(df), "(expected 1156)")
print("lapsed:", int(df["lapsed"].sum()), "=", round(df["lapsed"].mean() * 100, 2),
      "% (expected 443, 38.32%)")
print("train/test rows:", len(X_train), len(X_test), "(expected 924, 232)")

base = np.zeros(len(y_test), dtype=int)
print("baseline accuracy:", round(accuracy_score(y_test, base) * 100, 1),
      "% | lapsed F1:", f1_score(y_test, base, zero_division=0),
      "(expected 61.6%, 0)")

assert not df[NUM + CAT].isna().any().any(), "missing values found"
print("no feature column contains a missing value: OK")

pre = make_preprocessor().fit(X_train)          # fitted on TRAIN rows only
i = NUM.index("frequency")
print("scaler mean of frequency (train only):", round(pre.named_transformers_["num"].mean_[i], 4))
print("full-cohort mean of frequency:        ", round(df["frequency"].mean(), 4))