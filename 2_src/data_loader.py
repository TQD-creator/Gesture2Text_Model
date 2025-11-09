# In file: 2_src/data_loader.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(train_csv, valid_csv, test_csv, labels_to_remove=['DD']):
    """
    Loads, filters, splits, and scales all datasets.
    """
    # Read files
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)
    df_test = pd.read_csv(test_csv)
    
    # Filter (like you wanted for 'DD')
    df_train = df_train[~df_train['label'].isin(labels_to_remove)]
    df_valid = df_valid[~df_valid['label'].isin(labels_to_remove)]
    df_test = df_test[~df_test['label'].isin(labels_to_remove)]
    
    # Split X/y
    X_train, y_train = df_train.drop('label', axis=1), df_train['label']
    X_valid, y_valid = df_valid.drop('label', axis=1), df_valid['label']
    X_test, y_test = df_test.drop('label', axis=1), df_test['label']
    
    # Scale data
    scaler = StandardScaler().fit(X_train)
    
    # Transform
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs('../4_models/', exist_ok=True)
    joblib.dump(scaler, '../4_models/scaler.pkl')
    
    print("Data loaded, filtered, and scaled. Scaler saved.")
    
    return (X_train_scaled, y_train, 
            X_valid_scaled, y_valid, 
            X_test_scaled, y_test)