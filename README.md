# Predicting Protein Thermostability from Sequence-Derived Features

A complete end-to-end machine learning pipeline that predicts whether a protein 
is thermostable or mesostable using physicochemical features computed directly 
from its amino acid sequence — achieving **AUROC = 0.9815** with a tuned 
Random Forest classifier.

---

## Project Overview

Thermostable proteins — proteins that maintain their structure and function at 
high temperatures — are of fundamental importance in enzyme engineering, 
industrial biotechnology, and drug design. Understanding what makes a protein 
thermostable from its sequence alone is a central question in computational 
protein biology.

This project builds a machine learning classifier that answers:

> **Can we predict whether a protein is thermostable or mesostable 
> from its amino acid sequence alone?**

Using the **Meltome Atlas** (Jarzab et al. 2020, *Nature Methods*) — a 
large-scale dataset of experimentally measured protein melting temperatures 
across 13 organisms — we engineered 26 physicochemical features from raw 
protein sequences and trained three classical ML classifiers, achieving 
near-perfect discrimination between thermostable and mesostable proteins.

---

## Key Results

| Model | AUROC | Accuracy | F1 Score |
|-------|-------|----------|----------|
| Logistic Regression | 0.7827 | 0.7173 | 0.7629 |
| Support Vector Machine | 0.9296 | 0.8642 | 0.8799 |
| **Random Forest (default)** | **0.9801** | **0.9301** | **0.9367** |
| **Random Forest (tuned)** | **0.9815** | **0.9367** | **0.9425** |

**Best model:** Tuned Random Forest (300 trees, log2 features)  
**Evaluation:** 5-fold stratified cross-validation + held-out test set (n=7,200)

---

## Biological Findings

SHAP (SHapley Additive exPlanations) analysis revealed that the Random Forest 
learned biologically meaningful patterns — not spurious correlations:

| Rank | Feature | SHAP Importance | Biological meaning |
|------|---------|----------------|-------------------|
| 1 | Molecular weight | 0.0794 | Compact proteins → thermostable |
| 2 | GRAVY score | 0.0464 | Tight hydrophobic core → thermostable |
| 3 | Glutamine (AA_Q) | 0.0377 | Low glutamine → avoids deamidation at high T |
| 4 | Glycine (AA_G) | 0.0350 | Low glycine → less conformational flexibility |
| 5 | Serine (AA_S) | 0.0299 | Low serine → less chain flexibility |
| 6 | Asparagine (AA_N) | 0.0282 | Low asparagine → avoids deamidation at high T |

These findings are fully consistent with known mechanisms of thermal adaptation 
in thermophilic organisms — validating that the model learned real protein 
biochemistry from sequence data alone.

---

## Dataset

**Source:** Meltome Atlas — Jarzab et al. (2020), *Nature Methods*, 17, 495–503  
**Download:** [FLIP GitHub Repository](https://github.com/J-SNACKKB/FLIP/tree/main/splits/meltome)

| Statistic | Value |
|-----------|-------|
| Raw proteins | 201,283 |
| Thermostable (Tm > 60°C) | 20,051 |
| Mesostable (Tm < 45°C) | 16,049 |
| Grey zone dropped (45–60°C) | 165,183 |
| **Final dataset** | **35,999 proteins** |

**Labelling:**
- Label `1` = Thermostable (Tm > 60°C)
- Label `0` = Mesostable (Tm < 45°C)
- Grey zone (45–60°C) excluded — biologically ambiguous

---

## Features

26 physicochemical features computed from raw amino acid sequences using 
**Biopython's ProtParam module**:

**Amino Acid Composition (20 features)**
Fraction of each of the 20 standard amino acids in the sequence.

**Global Physicochemical Properties (6 features)**

| Feature | Description |
|---------|-------------|
| `mol_weight` | Total molecular mass (Daltons) |
| `isoelectric` | Isoelectric point (pH at zero net charge) |
| `gravy` | Grand Average of Hydropathy score |
| `instability` | Predicted chemical instability index |
| `aromaticity` | Fraction of aromatic residues (F, W, Y) |
| `aliphatic` | Aliphatic index — Ikai (1980) formula |

---

---

## Reproducing the Results

### 1. Clone the repository
```bash
git clone https://github.com/maulibhavsar16/protein-thermo-ml.git
cd protein-thermo-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the raw data
Follow the instructions in `notebooks/00_data_acquisition.ipynb` to download 
the Meltome Atlas FASTA file from the FLIP GitHub repository and place it in 
`data/raw/`.

### 4. Run the pipeline
```bash
# Parse and clean the dataset
python3 src/01_parse_data.py

# Compute physicochemical features
python3 src/02_feature_engineering.py

# Run EDA
python3 src/03_eda.py

# Train and evaluate ML models
python3 src/04_ml_models.py

# Hyperparameter tuning (warning: ~40 minutes)
python3 src/05_hyperparameter_tuning.py

# SHAP interpretation (warning: ~7 minutes for SHAP computation)
python3 src/06_shap_interpretation.py
```

Or explore interactively by running the numbered notebooks in order.

---

---

## References

1. Jarzab A. et al. (2020). Meltome atlas — thermal proteome stability 
   across the tree of life. *Nature Methods*, 17, 495–503.

2. Dallago C. et al. (2022). FLIP: Benchmark tasks in fitness landscape 
   inference for proteins. *Cell Systems*.

3. Lundberg S.M. & Lee S.I. (2017). A unified approach to interpreting 
   model predictions. *NeurIPS*.

4. Ikai A. (1980). Thermostability and aliphatic index of globular proteins. 
   *Journal of Biochemistry*, 88(6), 1895–1898.

5. Shapley L.S. (1953). A value for n-person games. 
   *Contributions to the Theory of Games*, 2, 307–317.

---

## Author

**Bhavsar Mauli Chirag**  
MSc Bioinformatics (2025)  

---

## Acknowledgements

Dataset: Meltome Atlas (Jarzab et al. 2020, *Nature Methods*)  
ML benchmark: FLIP repository (Dallago et al. 2022, *Cell Systems*)

