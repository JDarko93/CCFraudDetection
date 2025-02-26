import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import logging
import json
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fraud_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fraud_detection")

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# Ensure directories exist
ensure_dir("logs")
ensure_dir("results")

def load_model():
    """Load the trained model"""
    try:
        # Load the model (use Random Forest as it likely performed better)
        model = joblib.load('models/credit_card_fraud_random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        logger.info("Model loaded successfully")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_data():
    """Load new transaction data"""
    try:
        # Get list of all CSV files in new_transactions directory
        data_dir = "data/new_transactions"
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning("No new transaction files found")
            return None
            
        # Process each file
        all_data = []
        for file in csv_files:
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            all_data.append(df)
            logger.info(f"Loaded file: {file}")
            
        # Combine all data into a single DataFrame
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total transactions loaded: {combined_data.shape[0]}")
        return combined_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data, scaler):
    """Preprocess the data to match the model's expectations"""
    try:
        # Keep track of transaction IDs or other metadata
        if 'transaction_id' in data.columns:
            transaction_ids = data['transaction_id'].values
        else:
            # Create transaction IDs if they don't exist
            transaction_ids = [f"tx_{i}" for i in range(data.shape[0])]
        
        # Select only the features used during training
        expected_features = scaler.feature_names_in_
        # Filter columns if necessary
        data_features = data[expected_features] if all(col in data.columns for col in expected_features) else data
        
        # Scale the features
        scaled_data = scaler.transform(data_features)
        
        logger.info("Data preprocessing completed")
        return scaled_data, transaction_ids
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def predict_fraud(model, data):
    """Use the model to predict fraudulent transactions"""
    try:
        # Get fraud probabilities
        fraud_probs = model.predict_proba(data)[:, 1]
        # Get binary predictions (0 = normal, 1 = fraud)
        predictions = model.predict(data)
        
        logger.info(f"Predictions generated for {len(predictions)} transactions")
        logger.info(f"Detected {sum(predictions)} potentially fraudulent transactions")
        
        return predictions, fraud_probs
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise

def save_results(transaction_ids, predictions, probabilities, threshold=0.5):
    """Save prediction results to file"""
    try:
        results = pd.DataFrame({
            'transaction_id': transaction_ids,
            'fraud_prediction': predictions,
            'fraud_probability': probabilities
        })
        
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to CSV
        results_file = f"results/fraud_detection_results_{timestamp}.csv"
        results.to_csv(results_file, index=False)
        
        # Group results for summary reporting
        summary = {
            'timestamp': timestamp,
            'total_transactions': len(predictions),
            'flagged_transactions': int(sum(predictions)),
            'fraud_rate': float(sum(predictions) / len(predictions)),
            'high_risk_transactions': int(sum(probabilities > threshold))
        }
        
        # Save summary to JSON
        summary_file = f"results/fraud_detection_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Create alerts file for high-risk transactions
        high_risk = results[results['fraud_probability'] > threshold]
        if not high_risk.empty:
            high_risk_file = f"results/high_risk_transactions_{timestamp}.csv"
            high_risk.to_csv(high_risk_file, index=False)
            
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Summary saved to {summary_file}")
        
        # Archive processed files (optional)
        # [Code to move processed files to an archive directory]
        
        return results_file, summary_file
    
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main():
    """Main pipeline execution function"""
    try:
        # Make sure output directories exist
        ensure_dir("results")
        
        # Load the model
        model, scaler = load_model()
        
        # Load new data
        data = load_data()
        if data is None or data.empty:
            logger.warning("No data to process. Exiting.")
            return
            
        # Preprocess data
        processed_data, transaction_ids = preprocess_data(data, scaler)
        
        # Make predictions
        predictions, probabilities = predict_fraud(model, processed_data)
        
        # Save results
        results_file, summary_file = save_results(transaction_ids, predictions, probabilities)
        
        logger.info("Fraud detection pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()