# 06_shap_interpretation.py
# Purpose: SHAP interpretation of the tuned Random Forest model.
# Produces four SHAP visualisation plots connecting ML predictions
# back to protein biochemistry.
#
# Note: SHAP computation on 1,000 proteins takes ~7 minutes.
# Pre-computed SHAP values are saved in data/processed/06_shap_values.csv
# Run this script only if you need to regenerate the SHAP values.

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import shap
import os
import time

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# ── File paths ────────────────────────────────────────────────────────────────
DATA_PATH       = "data/processed/03_feature_matrix.csv"
SHAP_PATH       = "data/processed/06_shap_values.csv"
SAMPLE_INFO     = "data/processed/06_shap_sample_info.csv"
X_SAMPLE_PATH   = "data/processed/06_X_sample.csv"
MODEL_PATH      = "models/random_forest_tuned.pkl"
FIGURES_DIR     = "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns
                if c not in ['protein_id', 'tm', 'label']]
X = df[feature_cols].values
y = df['label'].values

# Load tuned model
with open(MODEL_PATH, 'rb') as f:
    pipeline = pickle.load(f)
rf_model = pipeline.named_steps['model']
scaler   = pipeline.named_steps['scaler']
X_scaled_df = pd.DataFrame(scaler.transform(X), columns=feature_cols)

print(f"Data loaded: {X.shape}")

# ── Compute SHAP values ───────────────────────────────────────────────────────
# Check if pre-computed SHAP values exist
if os.path.exists(SHAP_PATH):
    print("\nLoading pre-computed SHAP values from disk...")
    shap_vals   = pd.read_csv(SHAP_PATH).values
    X_sample    = pd.read_csv(X_SAMPLE_PATH)
    sample_info = pd.read_csv(SAMPLE_INFO)
    y_sample    = sample_info['label'].values
    print(f"Loaded SHAP values: {shap_vals.shape}")
else:
    print("\nComputing SHAP values for 1,000 proteins (~7 minutes)...")
    np.random.seed(42)
    thermo_idx    = np.where(y == 1)[0]
    meso_idx      = np.where(y == 0)[0]
    sample_thermo = np.random.choice(thermo_idx, size=550, replace=False)
    sample_meso   = np.random.choice(meso_idx,   size=450, replace=False)
    sample_idx    = np.concatenate([sample_thermo, sample_meso])
    np.random.shuffle(sample_idx)

    X_sample  = X_scaled_df.iloc[sample_idx].reset_index(drop=True)
    y_sample  = y[sample_idx]

    start    = time.time()
    explainer = shap.TreeExplainer(rf_model)
    shap_raw = explainer.shap_values(X_sample, check_additivity=False)
    elapsed  = time.time() - start
    print(f"Done in {elapsed:.1f} seconds!")

    shap_vals = shap_raw[:, :, 1]

    # Save to disk
    pd.DataFrame(shap_vals, columns=feature_cols).to_csv(SHAP_PATH, index=False)
    X_sample.to_csv(X_SAMPLE_PATH, index=False)
    pd.DataFrame({'sample_index': sample_idx,
                  'label': y_sample}).to_csv(SAMPLE_INFO, index=False)
    print("SHAP values saved to disk.")

# ── Plot 1: Beeswarm Summary Plot ─────────────────────────────────────────────
print("\nGenerating Plot 1: Beeswarm summary plot...")

plt.figure(figsize=(10, 10))
shap.summary_plot(shap_vals, X_sample,
                  feature_names=feature_cols,
                  plot_type='dot',
                  max_display=26,
                  show=False)
plt.title('SHAP Beeswarm Summary Plot\n'
          'Feature contributions to thermostability prediction (n=1,000 proteins)',
          fontsize=12, fontweight='500', pad=15)
plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, 'phase6_shap_beeswarm.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {save_path}")

# ── Plot 2: Feature Importance Bar Chart ──────────────────────────────────────
print("\nGenerating Plot 2: Feature importance bar chart...")

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
importance_df = pd.DataFrame({
    'feature'   : feature_cols,
    'importance': mean_abs_shap
}).sort_values('importance', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 9))
colors = ['tomato'     if v > 0.04 else
          'darkorange' if v > 0.02 else
          'steelblue'
          for v in importance_df['importance']]

bars = ax.barh(range(len(importance_df)),
               importance_df['importance'],
               color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(importance_df)))
ax.set_yticklabels(importance_df['feature'], fontsize=10)
ax.invert_yaxis()

for i, (val, bar) in enumerate(zip(importance_df['importance'], bars)):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)

ax.set_xlabel('Mean |SHAP value|', fontsize=11)
ax.set_title('SHAP Feature Importance\n'
             'Mean absolute SHAP value across 1,000 proteins',
             fontsize=12, fontweight='500')

legend_elements = [
    mpatches.Patch(facecolor='tomato',     label='High (> 0.04)'),
    mpatches.Patch(facecolor='darkorange', label='Moderate (0.02–0.04)'),
    mpatches.Patch(facecolor='steelblue',  label='Low (< 0.02)')
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, 'phase6_shap_importance.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {save_path}")

# ── Plot 3: Dependence Plots ───────────────────────────────────────────────────
print("\nGenerating Plot 3: Dependence plots...")

top_features = ['mol_weight', 'gravy', 'AA_Q', 'AA_G']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    ax         = axes[i]
    feat_idx   = feature_cols.index(feature)
    feat_vals  = X_sample[feature].values
    feat_shap  = shap_vals[:, feat_idx]

    scatter = ax.scatter(feat_vals, feat_shap,
                         c=feat_shap, cmap='coolwarm',
                         alpha=0.6, s=15, edgecolors='none')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    z = np.polyfit(feat_vals, feat_shap, 1)
    p = np.poly1d(z)
    x_line = np.linspace(feat_vals.min(), feat_vals.max(), 100)
    ax.plot(x_line, p(x_line), color='black', linewidth=1.5,
            alpha=0.8, label='Trend')

    plt.colorbar(scatter, ax=ax, label='SHAP value')
    ax.set_xlabel(f'{feature} value', fontsize=10)
    ax.set_ylabel(f'SHAP value for {feature}', fontsize=10)
    ax.set_title(feature, fontsize=11, fontweight='500')
    ax.legend(fontsize=8)

plt.suptitle('SHAP Dependence Plots — Top 4 Features',
             fontsize=12, fontweight='500', y=1.01)
plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, 'phase6_shap_dependence.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {save_path}")

# ── Plot 4: Waterfall Plots ────────────────────────────────────────────────────
print("\nGenerating Plot 4: Waterfall plots...")

X_scaled_sample = pipeline.named_steps['scaler'].transform(X_sample)
y_probs         = rf_model.predict_proba(X_scaled_sample)[:, 1]
base_rate       = y_sample.mean()
thermo_idx_w    = np.argmax(y_probs)
meso_idx_w      = np.argmin(y_probs)

def plot_waterfall(shap_row, feature_names, base_rate,
                   final_pred, title, ax, top_n=10):
    sorted_idx   = np.argsort(np.abs(shap_row))[::-1][:top_n]
    sorted_vals  = shap_row[sorted_idx][::-1]
    sorted_names = [feature_names[i] for i in sorted_idx][::-1]
    cumulative   = base_rate + np.cumsum(sorted_vals)
    starts       = np.concatenate([[base_rate], cumulative[:-1]])
    colors       = ['tomato' if v > 0 else 'steelblue' for v in sorted_vals]

    bars = ax.barh(range(top_n), sorted_vals, left=starts,
                   color=colors, edgecolor='white',
                   linewidth=0.5, alpha=0.85)
    for i, (val, start) in enumerate(zip(sorted_vals, starts)):
        ax.text(start + val/2, i, f'{val:+.3f}',
                ha='center', va='center',
                fontsize=7, color='white', fontweight='500')

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.axvline(x=base_rate, color='gray', linestyle='--',
               linewidth=1, alpha=0.7,
               label=f'Base rate = {base_rate:.3f}')
    ax.axvline(x=final_pred, color='black', linestyle='-',
               linewidth=1.5, alpha=0.9,
               label=f'Prediction = {final_pred:.3f}')
    ax.set_xlabel('Predicted probability (thermostable)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='500')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plot_waterfall(shap_vals[thermo_idx_w], feature_cols, base_rate,
               y_probs[thermo_idx_w],
               f'Most Thermostable Protein\n'
               f'(Predicted: {y_probs[thermo_idx_w]:.3f}, '
               f'True: {y_sample[thermo_idx_w]})',
               axes[0])
plot_waterfall(shap_vals[meso_idx_w], feature_cols, base_rate,
               y_probs[meso_idx_w],
               f'Most Mesostable Protein\n'
               f'(Predicted: {y_probs[meso_idx_w]:.3f}, '
               f'True: {y_sample[meso_idx_w]})',
               axes[1])

plt.suptitle('SHAP Waterfall Plots — Individual Protein Explanations',
             fontsize=12, fontweight='500', y=1.02)
plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, 'phase6_shap_waterfall.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {save_path}")

print("\nPhase 6 complete!")
print("All SHAP figures saved to figures/ directory.")