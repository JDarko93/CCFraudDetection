# Credit Card Fraud Detection Pipeline

This repository contains an automated pipeline for detecting fraudulent credit card transactions using machine learning.

## Setup Instructions

1. Clone this repository to your local machine.
2. Install the required dependencies:
pip install -r requirements.txt

3. Run the model training script to create the model files:
python save_model.py

4. Create the following directories if they don't exist:
- `data/new_transactions` (where new transaction files will be placed)
- `logs` (for pipeline logs)
- `results` (for detection results)
- `models` (for trained models)

## Using the Pipeline

### Manually Running the Pipeline

To manually process new transaction data:

1. Place new transaction CSV files in the `data/new_transactions` directory.
2. Run the pipeline:
python fraud_detection_pipeline.py

3. View the results in the `results` directory.

### GitHub Actions Automated Pipeline

The pipeline will automatically run:
- Every day at midnight (UTC)
- When new transaction files are pushed to the `data/new_transactions` directory
- When manually triggered through the GitHub Actions interface

## Results Format

The pipeline generates three types of output files:

1. **Detailed Results** (`fraud_detection_results_[timestamp].csv`): Contains predictions for all transactions.
2. **Summary Report** (`fraud_detection_summary_[timestamp].json`): Contains aggregated statistics.
3. **High-Risk Transactions** (`high_risk_transactions_[timestamp].csv`): Lists transactions with high fraud probability.

