# 01_parse_data.py
# Purpose: Parse the meltome FASTA file and create a clean CSV
# ready for feature engineering.
# The FASTA already contains both sequences AND melting points
# in the header — so we don't need the JSON at all.

import pandas as pd
from Bio import SeqIO
import os

# ── File paths ────────────────────────────────────────────────────────────────
FASTA_PATH = "data/raw/full_dataset_sequences.fasta"
OUT_PATH   = "data/processed/02_dataset_clean.csv"

# ── Step 1: Parse the FASTA file ──────────────────────────────────────────────
print("Parsing FASTA file...")
records = []

for record in SeqIO.parse(FASTA_PATH, "fasta"):
    # Extract melting point from description line
    # e.g. "A0A023T4K3_Caenorhabditis_elegans_lysate MELTING_POINT=37.96"
    tm = None
    for part in record.description.split():
        if part.startswith("MELTING_POINT="):
            tm = float(part.split("=")[1])

    # Skip if no melting point found
    if tm is None:
        continue

    records.append({
        "protein_id" : record.id,
        "sequence"   : str(record.seq),
        "tm"         : tm
    })

df = pd.DataFrame(records)
print(f"  Total records parsed: {len(df):,}")
print(f"  Tm range: {df.tm.min():.1f} – {df.tm.max():.1f} °C")

# ── Step 2: Assign binary labels ──────────────────────────────────────────────
# Thermostable : Tm > 60°C  → label 1
# Mesostable   : Tm < 45°C  → label 0
# Grey zone    : 45–60°C    → dropped (biologically ambiguous)

df_thermo = df[df["tm"] > 60].copy()
df_thermo["label"] = 1

df_meso = df[df["tm"] < 45].copy()
df_meso["label"] = 0

df_final = pd.concat([df_thermo, df_meso]).reset_index(drop=True)

print(f"\nAfter labelling and dropping grey zone:")
print(f"  Thermostable (label=1): {(df_final.label==1).sum():,}")
print(f"  Mesostable   (label=0): {(df_final.label==0).sum():,}")
print(f"  Grey zone dropped     : {(df.tm.between(45,60)).sum():,}")
print(f"  Total kept            : {len(df_final):,}")

# ── Step 3: Save ──────────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
df_final.to_csv(OUT_PATH, index=False)
print(f"\nSaved to: {OUT_PATH}")
print("Phase 1 complete!")