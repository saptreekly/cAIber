import os
import pandas as pd
import logging
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Set up logging
log_dir = '/Users/jackweekly/Desktop/Caiber/logs'
log_file = os.path.join(log_dir, 'evaluate_model.log')
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

if __name__ == "__main__":
    input_folder = "/Users/jackweekly/Desktop/Caiber/data/reduced_dimension"
    model_folder = "/Users/jackweekly/Desktop/Caiber/data/models"
    
    logging.info("Loading data")
    data = load_data(input_folder)
    X = data.drop(columns=['EventID', 'Channel', 'Computer', 'Provider', 'TimeCreated'])
    y = data['EventID']  # Assuming EventID is the target variable

    models = ['random_forest', 'gradient_boosting', 'svm']
    for model_name in models:
        model_path = os.path.join(model_folder, f"{model_name}.joblib")
        logging.info(f"Evaluating {model_name}")
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        logging.info(f"Accuracy of {model_name}: {accuracy_score(y, y_pred)}")
        logging.info(f"Classification Report for {model_name}:\n{classification_report(y, y_pred)}")