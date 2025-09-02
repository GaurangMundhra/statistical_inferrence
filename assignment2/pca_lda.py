import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_comparison(file):
    df = pd.read_csv(file.name)

    df = df.dropna(axis=1, how="all")
    if "id" in df.columns: 
        df = df.drop(columns=["id"])

    if "diagnosis" in df.columns:
        y = df["diagnosis"]
        X = df.drop(columns=["diagnosis"])
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    results = {}

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results["Original"] = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    results["PCA"] = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ]

    fig_pca, ax = plt.subplots(figsize=(6,5))
    ax.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.6)
    ax.set_title("PCA (2D Projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()

    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y)
    X_train_lda, X_test_lda, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)
    clf.fit(X_train_lda, y_train)
    y_pred = clf.predict(X_test_lda)
    results["LDA"] = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ]

    fig_lda, ax2 = plt.subplots(figsize=(6,5))
    ax2.scatter(X_lda, np.zeros_like(X_lda), c=y, cmap="coolwarm", alpha=0.6)
    ax2.set_title("LDA (1D Projection)")
    ax2.set_xlabel("LD1")
    ax2.set_yticks([])
    plt.tight_layout()

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    df_results = pd.DataFrame(results, index=metrics)

    fig_bar, ax3 = plt.subplots(figsize=(8,6))
    df_results.plot(kind="bar", ax=ax3)
    ax3.set_title("Comparison of Original, PCA, and LDA")
    ax3.set_ylabel("Score")
    ax3.set_ylim(0, 1.05)
    plt.tight_layout()

    return fig_pca, fig_lda, fig_bar

demo = gr.Interface(
    fn=run_comparison,
    inputs=gr.File(file_types=[".csv"], label="Upload CSV Dataset"),
    outputs=[
        gr.Plot(label="PCA Visualization"),
        gr.Plot(label="LDA Visualization"),
        gr.Plot(label="Comparison of Metrics")
    ],
    title="PCA vs LDA Dimensionality Reduction",
    description="Upload a dataset and compare PCA, LDA, and Original Logistic Regression performance."
)

if __name__ == "__main__":
    demo.launch()
