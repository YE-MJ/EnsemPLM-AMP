import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

feature_dir = './feature/independent/'
weight_base_dir = './weight/'
target_seed = 16
threshold = 0.42

fixed_features = ['esmc_600m', 'prot_t5_xl_bfd', 'prot_t5_xl_uniref50', 'esmc_300', 'esm2']
feature_list = fixed_features

weight_dir = os.path.join(weight_base_dir, f'seed{target_seed}')
ensemble_features_preds = []
y = None

for feature in feature_list:
    feature_file = os.path.join(feature_dir, f"{feature}.csv")
    if not os.path.exists(feature_file):
        continue

    df = pd.read_csv(feature_file)
    X = df.drop(['name', 'target'], axis=1)
    if y is None:
        y = df['target'].values

    fold_preds = []

    for fold_num in range(1, 11):
        model_path = os.path.join(weight_dir, f"SVM_{feature}_fold_{fold_num}.pt")
        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(X)[:, 1]
        else:
            pred = model.predict(X)
        fold_preds.append(pred)

    if len(fold_preds) == 0:
        continue

    fold_mean_pred = np.mean(fold_preds, axis=0)
    ensemble_features_preds.append(fold_mean_pred)

def evaluate_combination(weights, preds_list, y_true, threshold):
    weighted_preds = np.zeros_like(preds_list[0])
    for w, preds in zip(weights, preds_list):
        weighted_preds += w * preds

    preds_binary = (weighted_preds >= threshold).astype(int)

    acc = accuracy_score(y_true, preds_binary)
    sensitivity = recall_score(y_true, preds_binary)
    specificity = recall_score(y_true, preds_binary, pos_label=0)
    auc = roc_auc_score(y_true, weighted_preds)
    mcc = matthews_corrcoef(y_true, preds_binary)
    f1 = f1_score(y_true, preds_binary)

    return {
        'ACC': acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'AUC': auc,
        'MCC': mcc,
        'F1': f1
    }

fixed_weights = [0.06, 0.11, 0.32, 0.36, 0.15]

fixed_metrics = evaluate_combination(fixed_weights, ensemble_features_preds, y, threshold)

print(f"Evaluation on independent set using fixed weights:")
for feat, w in zip(fixed_features, fixed_weights):
    print(f"  {feat}: {w:.2f}")

print(f"\nMetrics with threshold = {threshold}")
for metric_name, value in fixed_metrics.items():
    print(f"  {metric_name}: {value:.4f}")
