# 02_feature_engineering.py
# Purpose: Convert raw protein sequences into a numerical feature matrix
# using physicochemical properties computed by Biopython's ProtParam module.

# ── Imports ───────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm   # progress bar — shows how many proteins processed
import os

# ── File paths ────────────────────────────────────────────────────────────────

# Input: the clean labelled dataset from Phase 1
IN_PATH  = "data/processed/02_dataset_clean.csv"

# Output: the feature matrix we will create in this script
OUT_PATH = "data/processed/03_feature_matrix.csv"

# ── Feature computation function ──────────────────────────────────────────────

def compute_features(sequence):
    """
    Takes a single protein sequence as a string (e.g. "MKKQTLSEW...")
    and returns a dictionary of 26 physicochemical features.
    
    We use a dictionary so each feature has a clear name that will 
    become a column name in our final table.
    """

    # Create a ProteinAnalysis object from the sequence
    # This is the main Biopython object that computes all properties
    analysis = ProteinAnalysis(sequence)
    
    # ── Feature Group 1: Amino Acid Composition (20 features) ─────────────────
    # amino_acids_percent() returns a dictionary like:
    # {'A': 0.082, 'C': 0.014, 'D': 0.054, ...} for all 20 amino acids
    aa_comp = analysis.amino_acids_percent
    
    # ── Feature Group 2: Global Physicochemical Properties (6 features) ───────
    
    # Molecular weight: total mass of the protein in Daltons
    mol_weight = analysis.molecular_weight()
    
    # Isoelectric point: pH at which protein has zero net charge
    isoelectric = analysis.isoelectric_point()
    
    # GRAVY: Grand Average of Hydropathy
    # Positive = hydrophobic, Negative = hydrophilic
    gravy = analysis.gravy()
    
    # Instability index: values > 40 suggest unstable protein
    instability = analysis.instability_index()
    
    # Aromaticity: fraction of Phe (F), Trp (W), Tyr (Y) in the sequence
    aromaticity = analysis.aromaticity()
    
    # Aliphatic index: computed manually using Ikai (1980) formula
    # Measures relative volume occupied by aliphatic side chains
    # Formula: AA_A + 2.9 * AA_V + 3.9 * (AA_I + AA_L)
    aliphatic = (aa_comp['A'] + 
                 2.9 * aa_comp['V'] + 
                 3.9 * (aa_comp['I'] + aa_comp['L']))
    
    # ── Build the final feature dictionary ────────────────────────────────────
    # We prefix amino acid features with 'AA_' so they are easy to identify
    features = {}
    
    # Add all 20 amino acid composition values
    for aa, percent in aa_comp.items():
        features[f'AA_{aa}'] = percent
    
    # Add the 6 global properties
    features['mol_weight']   = mol_weight
    features['isoelectric']  = isoelectric
    features['gravy']        = gravy
    features['instability']  = instability
    features['aromaticity']  = aromaticity
    features['aliphatic']    = aliphatic
    
    return features

# ── Main execution ────────────────────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv(IN_PATH)
print(f"  Loaded {len(df):,} proteins")

# ── Compute features for all proteins ─────────────────────────────────────────
# tqdm() wraps our loop and shows a progress bar
# It tells us how many proteins have been processed and estimates time remaining

print("\nComputing features for all proteins...")
print("(This will take 2-3 minutes — watch the progress bar!)\n")

all_features = []  # we will collect one dictionary per protein here

for sequence in tqdm(df['sequence'], desc="Computing features"):
    try:
        # Compute features for this protein
        features = compute_features(sequence)
        all_features.append(features)
    except Exception as e:
        # If a sequence causes an error (e.g. unusual characters)
        # we append None so we can identify and remove it later
        all_features.append(None)

# ── Build the feature matrix ───────────────────────────────────────────────────
print("\nBuilding feature matrix...")

# Find indices of failed proteins before building DataFrame
failed_indices = [i for i, f in enumerate(all_features) if f is None]
print(f"  Proteins that failed feature computation: {len(failed_indices)}")

# Remove None values from the list
all_features_clean = [f for f in all_features if f is not None]

# Also remove corresponding rows from original dataframe
df_clean = df.drop(index=failed_indices).reset_index(drop=True)

# Convert list of dictionaries into a DataFrame
feature_df = pd.DataFrame(all_features_clean)

# Check if any proteins failed feature computation
failed = feature_df.isnull().all(axis=1).sum()
print(f"  Proteins that failed feature computation: {failed}")

# Remove any failed rows
feature_df = feature_df.dropna(how='all')

# Add back the protein_id, tm and label columns from original dataset
feature_df.insert(0, 'protein_id', df['protein_id'].values[:len(feature_df)])
feature_df['tm']    = df['tm'].values[:len(feature_df)]
feature_df['label'] = df['label'].values[:len(feature_df)]

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nFeature matrix shape: {feature_df.shape}")
print(f"  Rows    : {feature_df.shape[0]:,} proteins")
print(f"  Columns : {feature_df.shape[1]} (1 protein_id + 26 features + 1 tm + 1 label)")
print(f"\nFeature columns:")
feature_cols = [c for c in feature_df.columns if c not in ['protein_id','tm','label']]
print(f"  {feature_cols}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
feature_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved to: {OUT_PATH}")
print("Phase 2 feature engineering complete!")