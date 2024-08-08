import os
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Set up logging
log_dir = '/Users/jackweekly/Desktop/Caiber/logs'
log_file = os.path.join(log_dir, 'train_model.log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if os.path.exists(log_file):
    os.remove(log_file)
logging.basicConfig(filename=log_file, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(input_folder):
    all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]
    df_list = [pd.read_csv(f) for f in all_files]
    return pd.concat(df_list, ignore_index=True)

def train_and_save_model(model, X_train, y_train, model_path):
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    input_folder = "/Users/jackweekly/Desktop/Caiber/data/reduced_dimension"
    output_folder = "/Users/jackweekly/Desktop/Caiber/data/models"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    logging.info("Loading data")
    data = load_data(input_folder)
    X = data.drop(columns=['EventID', 'Channel', 'Computer', 'Provider', 'TimeCreated'])
    y = data['EventID']  # Assuming EventID is the target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='linear', random_state=42)
    }

    for model_name, model in models.items():
        logging.info(f"Training {model_name}")
        model_path = os.path.join(output_folder, f"{model_name}.joblib")
        train_and_save_model(model, X_train, y_train, model_path)
    
    logging.info("All models trained and saved")