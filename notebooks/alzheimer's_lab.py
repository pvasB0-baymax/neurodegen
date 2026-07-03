import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (precision_recall_curve, classification_report, confusion_matrix, 
                             roc_auc_score, average_precision_score, balanced_accuracy_score,
                             matthews_corrcoef, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.impute import SimpleImputer

# --- Configuration ---
RANDOM_STATE = 42
DATA_PATH = "alzheimers_dataset_with_clinical.csv"
FEATURES = ["BIN1", "CLU", "PICALM", "ABCA7", "CD33", "APOE_E4_DOSAGE", "APOE_E2_DOSAGE", "APOE_E4_CARRIER", "AGE_AT_BASELINE", "PTGENDER_MALE"]

def get_data():
    df = pd.read_csv(DATA_PATH).dropna(subset=["AD_STATUS"])
    df = df[df["BASELINE_DX"] != "Dementia"].copy()
    
    # Preprocessing
    df["PTGENDER_MALE"] = (df["PTGENDER"] == "Male").astype(float)
    df["APOE_GENOTYPE"] = SimpleImputer(strategy='most_frequent').fit_transform(df[["APOE_GENOTYPE"]])
    
    X = pd.concat([df.drop(columns=["APOE_GENOTYPE"]), pd.get_dummies(df["APOE_GENOTYPE"], prefix="APOE")], axis=1)
    X = X[FEATURES + [c for c in X.columns if c.startswith("APOE_") and c not in FEATURES]]
    y = df["AD_STATUS"].astype(int)
    return X, y

def optimize(X_train, y_train):
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "scale_pos_weight": ratio,
            "random_state": RANDOM_STATE
        }
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        return average_precision_score(y_train, oof_proba)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=50)
    return {**study.best_params, "scale_pos_weight": ratio, "random_state": RANDOM_STATE}

def find_threshold(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thresh = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return thresh[np.argmax(f1[:-1])]

# --- Orchestration ---
X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)

best_params = optimize(X_train, y_train)
model = xgb.XGBClassifier(**best_params).fit(X_train, y_train)
threshold = find_threshold(model, X_train, y_train)

# --- Final Evaluation ---
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)

print(classification_report(y_test, pred))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=axes[0])
RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1])
PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[2])
plt.tight_layout()
plt.show()

# SHAP
shap.summary_plot(shap.TreeExplainer(model).shap_values(X_test), X_test)