# Unified Trust Metric System

A PyTorch-based prototype for detecting hallucinations in LLM-generated code and summaries using a multi-modal Trust Metric.

## Overview
This system processes a dataset of (question, code, summary) triplets, generates synthetic hallucinated variants, and trains a hybrid model to predict the trustworthiness of the code and summary.

## Project Structure
- `data_loading.py`: Loads `fyp dataset.xlsx`, generates synthetic data, extracts features, and creates `processed_trust_dataset.parquet`.
- `features.py`: Contains logic for tokenization, execution sandbox, summary entity extraction, and API checks.
- `model.py`: Defines the `HybridTrustModel` (CNN + MLP).
- `train.py`: Training loop and evaluation metrics.
- `config/config.yaml`: Hyperparameters and trust score weights.
- `ARCHITECTURE.md`: Detailed visual charts for the model and pipeline.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pandas openpyxl scikit-learn pyarrow pyyaml
   ```
2. Ensure you have the `code/fyp dataset.xlsx` file.

## Usage
1. **Generate Data**:
   ```bash
   python data_loading.py
   ```
   This will create `processed_trust_dataset.parquet`.

2. **Train and Evaluate**:
   ```bash
   python train.py
   ```
   This will train the model for 10 epochs (configurable) and report Precision, Recall, F1, and Trust Score Correlation.

## Methodology
- **Code Features**: Token-level features (type, normalized length, proxy probability) processed by a 1D CNN.
- **Execution Features**: Success status and error types from executing code (dry-run/safe exec).
- **Summary Features**: Entity overlap between code and summary.
- **Trust Score**: $w_1(1 - P_{code}) + w_2(1 - P_{summ}) + w_3 S_{api}$.
