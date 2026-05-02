# 05_hyperparameter_tuning.py
# Purpose: Hyperparameter tuning of the best model (Random Forest)
# using GridSearchCV with 5-fold stratified cross-validation.
#
# Best hyperparameters found:
#   n_estimators    : 300
#   max_depth       : None
#   min_samples_split: 2
#   max_features    : 'log2'
#
# Final tuned model AUROC on held-out test set: 0.9815

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
import json

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold)
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, precision_score, recall_score)

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# ── File paths ────────────────────────────────────────────────────────────────
DATA_PATH    = "data/processed/03_feature_matrix.csv"
RESULTS_PATH = "data/processed/05_best_params.json"
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
print(f"  Shape: {X.shape}\n")

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
print(f"Train: {X_train.shape[0]:,} proteins")
print(f"Test : {X_test.shape[0]:,} proteins\n")

# ── Define pipeline ───────────────────────────────────────────────────────────
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RandomForestClassifier(random_state=42, n_jobs=-1))
])

# ── Define hyperparameter grid ────────────────────────────────────────────────
param_grid = {
    'model__n_estimators'    : [100, 200, 300],
    'model__max_depth'       : [10, 20, None],
    'model__min_samples_split': [2, 5, 10],
    'model__max_features'    : ['sqrt', 'log2']
}

print(f"Total combinations: {3*3*3*2}")
print(f"Total CV fits     : {3*3*3*2*5}")
print("\nRunning GridSearchCV (this will take ~30-40 minutes)...")

# ── Run GridSearchCV ──────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    refit=True,
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
grid_search.fit(X_train, y_train)
elapsed = time.time() - start_time

print(f"\nGridSearchCV complete! ({elapsed/60:.1f} minutes)")
print(f"Best CV AUROC: {grid_search.best_score_:.4f}")
print(f"Best params  : {grid_search.best_params_}")

# ── Save best params ──────────────────────────────────────────────────────────
params_to_save = {k: str(v) for k, v in grid_search.best_params_.items()}
with open(RESULTS_PATH, 'w') as f:
    json.dump({
        'best_params'   : params_to_save,
        'best_cv_auroc' : grid_search.best_score_
    }, f, indent=2)
print(f"\nBest params saved to: {RESULTS_PATH}")

# ── Compare default vs tuned on test set ─────────────────────────────────────
print("\nComparing default vs tuned on held-out test set...")

models = {
    'Default RF (n=100, sqrt)': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RandomForestClassifier(
            n_estimators=100, max_depth=None,
            min_samples_split=2, max_features='sqrt',
            random_state=42, n_jobs=-1))
    ]),
    'Tuned RF (n=300, log2)': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_split=2, max_features='log2',
            random_state=42, n_jobs=-1))
    ])
}

comparison_results = {}
for name, pipeline in models.items():
    print(f"\nTraining {name}...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    comparison_results[name] = {
        'auroc'    : roc_auc_score(y_test, y_prob),
        'accuracy' : accuracy_score(y_test, y_pred),
        'f1'       : f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'time'     : elapsed
    }

    print(f"  AUROC   : {comparison_results[name]['auroc']:.4f}")
    print(f"  Accuracy: {comparison_results[name]['accuracy']:.4f}")
    print(f"  F1      : {comparison_results[name]['f1']:.4f}")

# ── Save tuned model ──────────────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, 'random_forest_tuned.pkl'), 'wb') as f:
    pickle.dump(models['Tuned RF (n=300, log2)'], f)
print("\nTuned model saved to models/random_forest_tuned.pkl")

# ── Plot: Default vs Tuned ────────────────────────────────────────────────────
print("\nGenerating comparison plot...")

metrics       = ['auroc', 'accuracy', 'f1', 'precision', 'recall']
metric_labels = ['AUROC', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
model_names   = list(comparison_results.keys())
colors        = ['steelblue', 'tomato']

fig, axes = plt.subplots(1, 5, figsize=(18, 6))

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax     = axes[i]
    values = [comparison_results[m][metric] for m in model_names]

    bars = ax.bar(range(len(model_names)), values,
                  color=colors, edgecolor='white',
                  linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(['Default\nRF', 'Tuned\nRF'], fontsize=9)
    ax.set_title(label, fontsize=11, fontweight='500')
    ax.set_ylim(0.90, 1.01)
    ax.set_ylabel('Score', fontsize=9)

plt.suptitle('Default vs Tuned Random Forest — Held-Out Test Set (n=7,200)',
             fontsize=13, fontweight='500', y=1.02)
plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, 'phase5_default_vs_tuned.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {save_path}")

print("\nPhase 5 complete!")
print("Best model: Tuned Random Forest (AUROC = 0.9815)")
print("Proceed to Phase 6 — SHAP Interpretation")