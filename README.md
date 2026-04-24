# Toxcity-Prediction

**AI-powered prediction of acute oral toxicity (LD50) from molecular structure.**

We benchmark six models — Ridge Regression, Random Forest, a DNN, a DNN-RF ensemble, a Message Passing Neural Network (MPNN), and a fine-tuned ChemBERTa transformer — on the [LD50_Zhu](https://tdcommons.ai/single_pred_tasks/tox/#ld50-zhu) dataset from Therapeutics Data Commons. All models are evaluated under both random and scaffold splits.

---

## Results

Test-set performance on z-score-normalized labels:

| Model             | Random R² | Random Spearman | Scaffold R² | Scaffold Spearman |
| ----------------- | --------- | --------------- | ----------- | ----------------- |
| Ridge Regression  | 0.509     | 0.658           | 0.178       | 0.523             |
| Random Forest     | 0.603     | 0.726           | 0.180       | 0.562             |
| DNN               | 0.625     | 0.738           | 0.262       | 0.613             |
| **Ensemble 70/30**| **0.635** | **0.742**       | **0.271**   | **0.625**         |
| MPNN              | 0.583     | 0.730           | 0.071       | 0.585             |
| ChemBERTa         | 0.536     | 0.671           | 0.173       | 0.500             |

The DNN-RF ensemble (0.7 × DNN + 0.3 × RF) wins on both splits. Fingerprint-based models consistently outperform the graph-based and transformer-based alternatives at this dataset scale (~5k training molecules) — consistent with the literature on data requirements for learned molecular representations.

---

## Installation

```bash
# Core scientific stack
pip install numpy pandas scikit-learn scipy matplotlib seaborn

# Cheminformatics + dataset
pip install rdkit
pip install PyTDC --no-deps

# Deep learning
pip install torch
pip install torch-geometric
pip install transformers datasets
```

A CUDA-capable GPU is strongly recommended for the DNN, MPNN, and ChemBERTa. The notebook was developed on Google Colab (T4/A100).

---

## Usage

Open `notebook.ipynb` and run cells top-to-bottom. The pipeline:

1. **Load data** via `tdc.single_pred.Tox('LD50_Zhu')` — downloads ~7,385 SMILES–LD50 pairs.
2. **Split** into random (80/10/10) and scaffold splits.
3. **Featurize** SMILES into 2048-bit ECFP4 fingerprints (radius 2) via RDKit.
4. **Normalize** labels via z-score (training statistics only).
5. **Train and evaluate** each of the six models.
6. **Generate visualizations** — model comparison, loss curves, predicted-vs-actual scatter plots, feature importance, generalization gap.

Total runtime: ~1–2 hours on a T4 GPU, dominated by ChemBERTa fine-tuning.

---

## Repository Structure

```
.
├── notebook.ipynb      # End-to-end pipeline: data → features → models → plots
├── ML_Project.pdf      # Project report
└── README.md
```

---

## Models

| Model        | Input representation       | Framework         |
| ------------ | -------------------------- | ----------------- |
| Ridge        | ECFP4 (2048 bits)          | scikit-learn      |
| Random Forest| ECFP4 (2048 bits)          | scikit-learn      |
| DNN          | ECFP4 (2048 bits)          | PyTorch           |
| Ensemble     | ECFP4 (2048 bits)          | PyTorch + sklearn |
| MPNN         | Molecular graph (8 atom + 4 bond features) | PyTorch Geometric |
| ChemBERTa    | SMILES tokens              | HuggingFace       |

**DNN**: 3 hidden layers `[1024, 512, 256, 128]` with BatchNorm, ReLU, dropout p=0.5. Adam (lr=1e-3, weight_decay=1e-3), ReduceLROnPlateau scheduler, early stopping (patience=20).

**MPNN**: 3 NNConv layers with edge-conditioned message passing, concatenated mean+max global pooling, 3-layer MLP head. Adam (lr=1e-3, weight_decay=1e-4).

**ChemBERTa**: Fine-tuned from `seyonec/ChemBERTa-zinc-base-v1` with differential learning rates (2e-5 for BERT layers, 1e-3 for regression head). AdamW, cosine annealing, mixed-precision training.

Hyperparameters for Ridge and RF are tuned via 5-fold CV (GridSearchCV / RandomizedSearchCV).

---

## Dataset

- **Source**: LD50_Zhu from Therapeutics Data Commons
- **Size**: 7,385 compounds
- **Task**: Regression — predict log(mol/kg) LD50 from SMILES
- **Splits**: Random 80/10/10, Scaffold (Bemis-Murcko) 80/10/10

---

## Key Findings

1. **Fingerprints beat learned representations at this scale.** ECFP4 encodes decades of pre-computed substructural chemistry; at ~5k training molecules, learned representations (MPNN, ChemBERTa) don't have enough signal to catch up.
2. **Scaffold split is hard.** All models lose 0.33–0.51 R² going from random to scaffold. MPNN collapses the hardest.
3. **Rankings survive better than absolute predictions.** Scaffold Spearman (~0.62 for the ensemble) stays meaningful even as R² degrades — practically useful for compound prioritization.

---

## Authors

- **Parthav Nuthalapati** — pipeline design, featurization, DNN, ensemble, MPNN, ChemBERTa, visualizations, generalization analysis
- **Zilu (Max) Zhou** — Random Forest tuning, preprocessing, label normalization, scaffold-split evaluation, comparative analysis, writeup

---

## References

Primary dataset and benchmark:

- Zhu, H. et al. (2009). *Quantitative structure-activity relationship modeling of rat acute toxicity by oral exposure.* Chem. Res. Toxicol. 22(12), 1913–1921.
- Huang, K. et al. (2021). *Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development.* NeurIPS Datasets and Benchmarks.

Representations and models:

- Rogers, D. & Hahn, M. (2010). *Extended-Connectivity Fingerprints.* JCIM 50(5), 742–754.
- Gilmer, J. et al. (2017). *Neural Message Passing for Quantum Chemistry.* ICML.
- Chithrananda, S. et al. (2020). *ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction.* arXiv:2010.09885.

See `ML_Project.pdf` for the full reference list.
