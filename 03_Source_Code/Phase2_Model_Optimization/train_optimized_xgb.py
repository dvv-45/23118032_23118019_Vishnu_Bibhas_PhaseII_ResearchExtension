import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold

# Ensure directories exist
os.makedirs("../../09_Trained_Models", exist_ok=True)
os.makedirs("../../07_Result_Images", exist_ok=True)
os.makedirs("../../08_Comparative_Results", exist_ok=True)

# 1. Load Vishnu's Handover Data
print("[INFO] Loading Phase II selected features...")
train_df = pd.read_csv("../../Vishnu_To_Bibhas/train_selected_phase2.csv")
test_df = pd.read_csv("../../Vishnu_To_Bibhas/test_selected_phase2.csv")

# Assume last column is the target label
X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

# 2. Optuna Objective Function for XGBoost
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'eval_metric': 'logloss',
        'random_state': 42,
        'tree_method': 'hist' # Faster training
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, average='weighted'))
        
    return np.mean(scores)

# 3. Run Optimization
print("[INFO] Starting Optuna Hyperparameter Tuning...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30) # Set to 30 for speed; increase if needed
best_params = study.best_params
print(f"[INFO] Best Params found: {best_params}")

# 4. Train Final Model & Calculate Latency
print("[INFO] Training final optimized model...")
final_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss', tree_method='hist')
final_model.fit(X_train, y_train)

start_time = time.perf_counter()
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
end_time = time.perf_counter()

latency_ms = ((end_time - start_time) / len(X_test)) * 1000

# 5. Evaluate Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"

print(f"\n--- Phase II Final Results ---")
print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print(f"ROC-AUC: {roc_auc} | Inference Latency per packet: {latency_ms:.4f} ms")

# 6. Save Model
joblib.dump(final_model, "../../09_Trained_Models/Phase2_Trained_Model.joblib")

# 7. Generate & Save Plots
print("[INFO] Generating Visualizations...")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Phase II Optimized Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("../../07_Result_Images/Confusion_Matrix_Phase2.png", dpi=300, bbox_inches='tight')
plt.close()

# ROC Curve
if y_pred_proba is not None:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Phase II')
    plt.legend(loc='lower right')
    plt.savefig("../../07_Result_Images/ROC_Curve_Phase2.png", dpi=300, bbox_inches='tight')
    plt.close()

# SHAP Explainability
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("../../07_Result_Images/SHAP_Summary_Phase2.png", dpi=300, bbox_inches='tight')
plt.close()

print("[SUCCESS] All Phase II downstream modeling complete and saved!")