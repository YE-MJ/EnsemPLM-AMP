import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

input_directory_path = "./feature/train/"
file_name = "esmc_600m.csv"
file_path = os.path.join(input_directory_path, file_name)

df = pd.read_csv(file_path)
X = df.drop(['name', 'target'], axis=1)
y = df['target']

params = {
    'C': 9.55,
    'degree': 4,
    'gamma': 0.74,
    'kernel': 'rbf'
}
seed = 16

seed_weight_dir = f"./weight/seed{seed}"
os.makedirs(seed_weight_dir, exist_ok=True)

fold_path = f"./folds/seed{seed}_fold_indices.pkl"

fold_indices = joblib.load(fold_path)

fold_metrics = []

for fold_index, (train_index, val_index) in enumerate(fold_indices, 1):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = SVC(**params, probability=True, random_state=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = accuracy_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)

    fold_metrics.append({
        'fold': fold_index,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'accuracy': accuracy,
        'mcc': mcc
    })

    model_filename = os.path.join(seed_weight_dir, f"SVM_esmc_600m_fold_{fold_index}.pt")
    joblib.dump(model, model_filename)

    print(f"[Seed {seed} - Fold {fold_index}] "
            f"Spec: {specificity:.4f}, Sens: {sensitivity:.4f}, "
            f"Acc: {accuracy:.4f}, MCC: {mcc:.4f}")

metrics_filename = os.path.join(seed_weight_dir, "SVM_esmc_600m_fold_metrics.txt")
with open(metrics_filename, 'w', encoding='utf-8') as f:
    for m in fold_metrics:
        f.write(f"Fold {m['fold']}:\n")
        f.write(f"  TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}, TP: {m['tp']}\n")
        f.write(f"  Specificity: {m['specificity']:.4f}, Sensitivity: {m['sensitivity']:.4f}\n")
        f.write(f"  Accuracy: {m['accuracy']:.4f}, MCC: {m['mcc']:.4f}\n\n")

