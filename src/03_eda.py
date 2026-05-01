# 03_eda.py
# Purpose: Exploratory Data Analysis (EDA) of the feature matrix
# We deeply understand our 26 features before building any ML models.
# This script produces all EDA figures saved to the figures/ directory.

# ── Imports ───────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# ── File paths ────────────────────────────────────────────────────────────────
DATA_PATH    = "data/processed/03_feature_matrix.csv"
FIGURES_DIR  = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading feature matrix...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# Separate feature columns from identifiers and target
# We will use feature_cols throughout the entire script
feature_cols = [c for c in df.columns 
                if c not in ['protein_id', 'tm', 'label']]

print(f"  Number of features: {len(feature_cols)}")
print(f"  Features: {feature_cols}")
print(f"\nData loaded successfully!")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Feature Distributions by Class
# ──────────────────────────────────────────────────────────────────────────────
# For each of our 26 features, we plot the distribution separately for
# thermostable (label=1) and mesostable (label=0) proteins.
#
# WHY: This tells us at a glance which features carry signal — features
# where the two distributions are clearly separated will be useful
# predictors. Features where they completely overlap will be weak.
# ──────────────────────────────────────────────────────────────────────────────

print("\nPlot 1: Feature distributions by class...")

# Separate the two classes into two dataframes for easy plotting
thermo = df[df['label'] == 1]   # thermostable proteins
meso   = df[df['label'] == 0]   # mesostable proteins

# We have 26 features — arrange them in a 6 rows × 5 columns grid
# (6 × 5 = 30 subplots, last 4 will be empty — that's fine)
n_cols = 5
n_rows = 6

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 18))

# .flatten() converts the 2D grid of axes into a simple 1D list
# so we can loop over them with a single index
axes = axes.flatten()

for i, feature in enumerate(feature_cols):
    ax = axes[i]

    # Plot mesostable distribution in blue
    ax.hist(meso[feature], bins=40, alpha=0.6,
            color='steelblue', label='Mesostable',
            edgecolor='none')

    # Plot thermostable distribution in red
    ax.hist(thermo[feature], bins=40, alpha=0.6,
            color='tomato', label='Thermostable',
            edgecolor='none')

    # Set title to feature name
    ax.set_title(feature, fontsize=10, fontweight='500')
    ax.set_xlabel('Value', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

# Hide the empty subplots (we have 30 slots but only 26 features)
for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Feature Distributions: Thermostable vs Mesostable Proteins',
             fontsize=14, fontweight='500', y=1.01)
plt.tight_layout()

# Save the figure
save_path = os.path.join(FIGURES_DIR, 'phase3_plot1_feature_distributions.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Feature Separation Ranking (Cohen's d)
# ──────────────────────────────────────────────────────────────────────────────
# For each feature, compute how well it separates the two classes
# using Cohen's d effect size metric.
# ──────────────────────────────────────────────────────────────────────────────

print("\nPlot 2: Feature separation ranking...")

thermo = df[df['label'] == 1]
meso   = df[df['label'] == 0]

effect_sizes = []
for feature in feature_cols:
    mean_thermo = thermo[feature].mean()
    mean_meso   = meso[feature].mean()
    std_all     = df[feature].std()
    cohens_d    = abs(mean_thermo - mean_meso) / std_all
    effect_sizes.append({
        'feature'  : feature,
        'cohens_d' : cohens_d
    })

effects_df = pd.DataFrame(effect_sizes).sort_values('cohens_d', ascending=False)
effects_df = effects_df.reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 9))

colors = ['tomato' if d > 0.4 else
          'darkorange' if d > 0.2 else
          'steelblue'
          for d in effects_df['cohens_d']]

bars = ax.barh(range(len(effects_df)),
               effects_df['cohens_d'],
               color=colors, edgecolor='white', linewidth=0.5)

ax.set_yticks(range(len(effects_df)))
ax.set_yticklabels(effects_df['feature'], fontsize=10)
ax.invert_yaxis()

for i, (val, bar) in enumerate(zip(effects_df['cohens_d'], bars)):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)

ax.axvline(0.2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(0.4, color='gray', linestyle='-',  linewidth=1, alpha=0.7)
ax.set_xlabel("Effect Size (Cohen's d)", fontsize=11)
ax.set_title("Feature Separation Ranking\n"
             "How well does each feature separate thermostable from mesostable proteins?",
             fontsize=12, fontweight='500')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='tomato',     label="Strong (d > 0.4)"),
    Patch(facecolor='darkorange', label="Moderate (0.2 < d ≤ 0.4)"),
    Patch(facecolor='steelblue',  label="Weak (d ≤ 0.2)")
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, 'phase3_plot2_feature_ranking.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {save_path}")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Correlation Heatmap
# ──────────────────────────────────────────────────────────────────────────────
# Compute Pearson correlation between all 26 features and the label.
# Reveals redundant features and features most correlated with the target.
# ──────────────────────────────────────────────────────────────────────────────

print("\nPlot 3: Correlation heatmap...")

corr_matrix = df[feature_cols + ['label']].corr()

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(corr_matrix,
            vmin=-1, vmax=1, center=0,
            cmap='coolwarm',
            annot=True,
            fmt='.1f',
            annot_kws={'size': 6},
            linewidths=0.3,
            ax=ax)

ax.set_title('Feature Correlation Matrix\n'
             'Pearson correlation between all 26 features and the label',
             fontsize=13, fontweight='500', pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, 'phase3_plot3_correlation_heatmap.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {save_path}")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 4 — PCA Scatter Plot
# ──────────────────────────────────────────────────────────────────────────────
# Compress 26 features into 2 dimensions using PCA and plot all
# 35,999 proteins coloured by class to visualise separability.
# ──────────────────────────────────────────────────────────────────────────────

print("\nPlot 4: PCA scatter plot...")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"  PC1 explains: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2 explains: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"  Total       : {sum(pca.explained_variance_ratio_)*100:.1f}%")

fig, ax = plt.subplots(figsize=(10, 8))

meso_mask   = df['label'].values == 0
thermo_mask = df['label'].values == 1

ax.scatter(X_pca[meso_mask,   0], X_pca[meso_mask,   1],
           c='steelblue', alpha=0.3, s=5,
           label=f'Mesostable (n={meso_mask.sum():,})')

ax.scatter(X_pca[thermo_mask, 0], X_pca[thermo_mask, 1],
           c='tomato', alpha=0.3, s=5,
           label=f'Thermostable (n={thermo_mask.sum():,})')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance explained)',
              fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance explained)',
              fontsize=11)
ax.set_title('PCA of Protein Feature Matrix\n'
             'Each dot represents one protein, coloured by thermostability class',
             fontsize=12, fontweight='500')
ax.legend(fontsize=10, markerscale=3)

plt.tight_layout()
save_path = os.path.join(FIGURES_DIR, 'phase3_plot4_pca.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {save_path}")

print("\nAll EDA plots saved successfully!")
print("Phase 3 complete!")