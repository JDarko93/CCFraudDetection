import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

def train_and_save_model():
    print("Loading data...")
    # Load your creditcard.csv data here
    credit_card_data = pd.read_csv('creditcard.csv')
    
    # Separate normal and fraudulent transactions
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]
    
    # Undersampling
    n_fraud = len(fraud)
    legit_sample = legit.sample(n=n_fraud, random_state=42)
    balanced_data = pd.concat([legit_sample, fraud], axis=0)
    
    # Split features and target
    X = balanced_data.drop('Class', axis=1)
    y = balanced_data['Class']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train the model
    print("Training Random Forest model...")
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)
    
    # Save the model and scaler
    print("Saving model and scaler...")
    joblib.dump(random_forest, 'models/credit_card_fraud_random_forest_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_and_save_model()