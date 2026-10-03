# Bank Marketing Capstone — From Raw Data to Real Predictions

End-to-end data science pipeline on the Bank Marketing dataset: SQL-based EDA, feature engineering,
K-Means customer segmentation, and a classification model to predict term-deposit subscription —
built for the TS Academy Data Science Capstone (Group 24).

## Project Overview

A bank ran a telemarketing campaign to sell term deposits. This project answers two business questions:

1. **What natural customer segments exist in the data?** (unsupervised clustering)
2. **Can we predict which customers will subscribe to a term deposit?** (supervised classification)

The notebook follows a full pipeline: load data → SQL exploration → EDA → feature engineering →
clustering → classification → imbalance handling → tie together clustering and classification results →
conclusions.

## Dataset

- **Source:** [Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset) (Kaggle, via `kagglehub`)
- **Size:** 11,162 rows, 17 columns
- **Target:** `deposit` — did the customer subscribe to a term deposit? (`yes` / `no`)
- **Class balance:** ~52.6% no / 47.4% yes (roughly balanced)

## Pipeline Summary

| Stage | What it does |
|---|---|
| **Data Foundation** | Loads data via `kagglehub`, pushes it into SQLite, runs 3 SQL queries, performs EDA, handles `unknown` categorical values |
| **Feature Engineering** | Builds `age_group`, `contacted_before`, `campaign_intensity_ratio`; encodes categoricals; scales numeric features |
| **Clustering** | K-Means (K=2–10), elbow + silhouette analysis, K=3 chosen, clusters profiled and named, visualized via PCA |
| **Classification** | Logistic Regression, Decision Tree, Random Forest vs. a stratified dummy baseline |
| **Imbalance Handling** | Class weighting, threshold tuning, and SMOTE compared against baseline |
| **Tie-In** | Checks whether unsupervised clusters line up with actual subscription rate |
| **Depth Track (Classification)** | Adds Gradient Boosting, 5-fold cross-validation, combined ROC curves, feature importance comparison |

## Key Results

- **Best model:** Random Forest, decision threshold tuned to 0.40
- **F1 score:** 0.701 | **ROC AUC:** 0.774
- Beats the required stratified-dummy baseline (F1 0.486 / ROC AUC 0.509) on both metrics
- No leakage: `duration` (a near-perfect predictor only known *after* a call happens) was identified and excluded from all modeling features
- Clustering surfaced a "warm returning customers" segment that subscribes at 66.8%, vs. 47.4% overall — confirming the unsupervised structure captures real signal

## Repository Structure

```
.
├── README.md
├── group_24_solomon_buba_tarfa.ipynb   # main notebook — full pipeline, run top to bottom
└── bank_marketing.db                   # generated SQLite database (created when the notebook runs)
```

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/tarfajr/bank-marketing-capstone.git
   cd bank-marketing-capstone
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn kagglehub
   ```
3. Set up a Kaggle API token (required for `kagglehub` to download the dataset):
   - Go to [kaggle.com](https://kaggle.com) → Settings → API → **Create New Token**
   - Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json` (`C:\Users\<you>\.kaggle\kaggle.json` on Windows)
4. Open the notebook and run **Kernel → Restart Kernel and Run All Cells**:
   ```bash
   jupyter notebook group_24_solomon_buba_tarfa.ipynb
   ```

## Tools & Libraries

`Python` · `pandas` · `numpy` · `scikit-learn` · `imbalanced-learn` · `matplotlib` · `seaborn` · `SQLite` · `kagglehub`

## Author

**Solomon Buba Tarfa** — Group 24, TS Academy Data Science Capstone
GitHub: [github.com/tarfajr](https://github.com/tarfajr)
