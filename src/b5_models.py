import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from lapse_features import get_split, make_preprocessor

df, X_train, X_test, y_train, y_test = get_split()

lr_pipe = Pipeline([("pre", make_preprocessor()),
                    ("clf", LogisticRegression(max_iter=1000))])
lr_grid = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]},
                       cv=5, scoring="f1").fit(X_train, y_train)

rf_pipe = Pipeline([("pre", make_preprocessor()),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_grid = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8],
                                 "clf__min_samples_leaf": [5, 10, 20]},
                       cv=5, scoring="f1").fit(X_train, y_train)

print("=== Tuning (cv=5, scoring=f1) ===")
print("LogReg best params:", lr_grid.best_params_, "| best CV F1:", round(lr_grid.best_score_, 3))
print("RandForest best params:", rf_grid.best_params_, "| best CV F1:", round(rf_grid.best_score_, 3))

print("\n=== Test set, lapsed class ===")
for name, g in [("Logistic Regression", lr_grid), ("Random Forest", rf_grid)]:
    pred = g.best_estimator_.predict(X_test)
    print(f"\n{name}")
    print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))
    print("accuracy :", round(accuracy_score(y_test, pred), 3))
    print("precision:", round(precision_score(y_test, pred, zero_division=0), 3))
    print("recall   :", round(recall_score(y_test, pred, zero_division=0), 3))
    print("F1       :", round(f1_score(y_test, pred, zero_division=0), 3))

best = {"C": lr_grid.best_params_["clf__C"],
        "max_depth": rf_grid.best_params_["clf__max_depth"],
        "min_samples_leaf": rf_grid.best_params_["clf__min_samples_leaf"]}
json.dump(best, open("outputs/best_params.json", "w"))
print("\nSaved outputs/best_params.json")