import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target   # 0 = Malignant, 1 = Benign

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=23)
}


def evaluate_model(model, X, y, cv):
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred))
        metrics["recall"].append(recall_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred))
        metrics["roc_auc"].append(roc_auc_score(y_test, y_prob))
    return metrics

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

results = {"KFold": {}, "StratifiedKFold": {}}

for name, model in models.items():
    results["KFold"][name] = evaluate_model(model, X, y, kf)
    results["StratifiedKFold"][name] = evaluate_model(model, X, y, skf)

summary_tables = {}
for scheme, res in results.items():
    rows = []
    for model, scores in res.items():
        avg_scores = {m: np.mean(v) for m, v in scores.items()}
        rows.append([model] + list(avg_scores.values()))
    summary_tables[scheme] = pd.DataFrame(
        rows,
        columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    )

    print(f"\n{scheme} Results")
    print(summary_tables[scheme].to_string(index=False, float_format="%.4f"))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

kf_class_counts = [np.bincount(y[test_idx], minlength=2) for _, test_idx in kf.split(X, y)]
sns.barplot(data=pd.DataFrame(kf_class_counts, columns=["Malignant", "Benign"]), ax=axes[0])
axes[0].set_title("KFold Class Distribution")

skf_class_counts = [np.bincount(y[test_idx], minlength=2) for _, test_idx in skf.split(X, y)]
sns.barplot(data=pd.DataFrame(skf_class_counts, columns=["Malignant", "Benign"]), ax=axes[1])
axes[1].set_title("StratifiedKFold Class Distribution")

plt.tight_layout()
plt.show()

for scheme, res in results.items():
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        plt.figure(figsize=(8, 5))
        data = {model: scores[metric] for model, scores in res.items()}
        sns.boxplot(data=pd.DataFrame(data))
        plt.title(f"{scheme} - {metric.capitalize()} across folds")
        plt.ylabel(metric.capitalize())
        plt.show()
