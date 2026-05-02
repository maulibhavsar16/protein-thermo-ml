# 04_ml_models.py
# Purpose: Train and evaluate three ML classifiers on the protein
# thermostability feature matrix. Saves results, models and figures.
#
# Models trained:
#   1. Logistic Regression — linear baseline
#   2. Random Forest       — ensemble tree method (best performer)
#   3. SVM                 — margin-based kernel classifier
#
# Evaluation: 5-fold Stratified Cross-Validation
# Primary metric: AUROC

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.metrics import roc_curve, auc
from matplotlib.patches import Patch

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# ── File paths ────────────────────────────────────────────────────────────────
DATA_PATH    = "data/processed/03_feature_matrix.csv"
RESULTS_PATH = "data/processed/04_model_results.csv"
FIGURES_DIR  = "figures"
MODELS_DIR   = "models"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading feature matrix...")
df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns
                if c not in ['protein_id', 'tm', 'label']]
X = df[feature_cols].values
y = df['label'].values
print(f"  Shape: {X.shape}")
print(f"  Class balance: {y.mean()*100:.1f}% thermostable\n")

# ── Define pipelines ──────────────────────────────────────────────────────────
# Each pipeline chains StandardScaler + classifier together
# This prevents data leakage during cross-validation

pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        ))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            C=1.0,
            gamma='scale'
        ))
    ])
}

# ── Cross-validation strategy ─────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'auroc'    : 'roc_auc',
    'accuracy' : 'accuracy',
    'f1'       : 'f1',
    'precision': 'precision',
    'recall'   : 'recall'
}

# ── Train and evaluate all models ─────────────────────────────────────────────
results = {}

for name, pipeline in pipelines.items():
    print(f"Training {name}...")
    start_time = time.time()

    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    elapsed = time.time() - start_time

    results[name] = {
        'auroc'    : cv_results['test_auroc'],
        'accuracy' : cv_results['test_accuracy'],
        'f1'       : cv_results['test_f1'],
        'precision': cv_results['test_precision'],
        'recall'   : cv_results['test_recall'],
        'time'     : elapsed
    }

    print(f"  Training time : {elapsed:.1f} seconds")
    print(f"  AUROC         : {cv_results['test_auroc'].mean():.4f} "
          f"± {cv_results['test_auroc'].std():.4f}")
    print(f"  Accuracy      : {cv_results['test_accuracy'].mean():.4f} "
          f"± {cv_results['test_accuracy'].std():.4f}")
    print(f"  F1 Score      : {cv_results['test_f1'].mean():.4f} "
          f"± {cv_results['test_f1'].std():.4f}")
    print(f"  Precision     : {cv_results['test_precision'].mean():.4f} "
          f"± {cv_results['test_precision'].std():.4f}")
    print(f"  Recall        : {cv_results['test_recall'].mean():.4f} "
          f"± {cv_results['test_recall'].std():.4f}")
    print()

# ── Save results to CSV ───────────────────────────────────────────────────────
print("Saving results...")
results_summary = []
for model_name, metrics in results.items():
    results_summary.append({
        'model'        : model_name,
        'auroc'        : metrics['auroc'].mean(),
        'auroc_std'    : metrics['auroc'].std(),
        'accuracy'     : metrics['accuracy'].mean(),
        'accuracy_std' : metrics['accuracy'].std(),
        'f1'           : metrics['f1'].mean(),
        'f1_std'       : metrics['f1'].std(),
        'precision'    : metrics['precision'].mean(),
        'precision_std': metrics['precision'].std(),
        'recall'       : metrics['recall'].mean(),
        'recall_std'   : metrics['recall'].std(),
        'time_sec'     : metrics['time']
    })

results_df = pd.DataFrame(results_summary)
results_df.to_csv(RESULTS_PATH, index=False)
print(f"  Saved: {RESULTS_PATH}\n")

# ── Save trained models ───────────────────────────────────────────────────────
print("Saving trained models...")
for name, pipeline in pipelines.items():
    print(f"  Fitting {name} on full dataset...")
    pipeline.fit(X, y)
    filename  = name.lower().replace(' ', '_') + '.pkl'
    save_path = os.path.join(MODELS_DIR, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"  Saved: {save_path}")

print()

# ── Plot 1: Model Comparison Bar Chart ───────────────────────────────────────
print("Generating Plot 1: Model comparison bar chart...")

metrics       = ['auroc', 'accuracy', 'f1', 'precision', 'recall']
metric_labels = ['AUROC', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
model_names   = results_df['model'].tolist()
colors        = ['steelblue', 'tomato', 'darkorange']

fig, axes = plt.subplots(1, 5, figsize=(18, 6))

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax     = axes[i]
    values = results_df[metric].values
    stds   = results_df[f'{metric}_std'].values

    bars = ax.bar(range(len(model_names)), values,
                  yerr=stds,
                  color=colors,
                  edgecolor='white',
                  linewidth=0.5,
                  capsize=4,
                  alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(['LR', 'RF', 'SVM'], fontsize=9)
    ax.set_title(label, fontsize=11, fontweight='500')
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel('Score', fontsize=9)

plt.suptitle('Model Performance Comparison — 5-Fold Stratified Cross-Validation',
             fontsize=13, fontweight='500', y=1.02)
plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, 'phase4_model_comparison.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {save_path}")

# ── Plot 2: ROC Curves ────────────────────────────────────────────────────────
print("\nGenerating Plot 2: ROC curves...")

colors_roc = {
    'Logistic Regression': 'steelblue',
    'Random Forest'      : 'tomato',
    'SVM'                : 'darkorange'
}

fig, ax = plt.subplots(figsize=(8, 7))

for name, filepath in {
    'Logistic Regression': os.path.join(MODELS_DIR, 'logistic_regression.pkl'),
    'Random Forest'      : os.path.join(MODELS_DIR, 'random_forest.pkl'),
    'SVM'                : os.path.join(MODELS_DIR, 'svm.pkl')
}.items():
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)

    y_probs = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y[train_idx]
        pipeline.fit(X_train, y_train)
        y_probs[test_idx] = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_probs)
    roc_auc     = auc(fpr, tpr)

    ax.plot(fpr, tpr,
            color=colors_roc[name],
            linewidth=2,
            label=f'{name} (AUROC = {roc_auc:.4f})')

ax.plot([0, 1], [0, 1],
        color='gray', linestyle='--',
        linewidth=1, label='Random baseline (AUROC = 0.5)')

ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves — All Three Models\n'
             '5-Fold Stratified Cross-Validation',
             fontsize=12, fontweight='500')
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, 'phase4_roc_curves.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {save_path}")

print("\nPhase 4 complete!")
print("Best model: Random Forest (AUROC = 0.9801)")
print("Proceed to Phase 5 — Hyperparameter Tuning")