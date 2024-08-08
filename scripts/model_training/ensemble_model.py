import os
import pandas as pd
import logging
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Set up logging
log_dir = '/Users/jackweekly/Desktop/Caiber/logs'
log_file = os.path.join(log_dir, 'ensemble_model.log')
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
    ensemble_model_path = os.path.join(model_folder, 'ensemble_model.joblib')
    
    logging.info("Loading data")
    data = load_data(input_folder)
    X = data.drop(columns=['EventID', 'Channel', 'Computer', 'Provider', 'TimeCreated'])
    y = data['EventID']  # Assuming EventID is the target variable

    models = {
        'random_forest': joblib.load(os.path.join(model_folder, 'random_forest.joblib')),
        'gradient_boosting': joblib.load(os.path.join(model_folder, 'gradient_boosting.joblib')),
        'svm': joblib.load(os.path.join(model_folder, 'svm.joblib'))
    }
    
    ensemble = VotingClassifier(estimators=[
        ('random_forest', models['random_forest']),
        ('gradient_boosting', models['gradient_boosting']),
        ('svm', models['svm'])
    ], voting='hard')
    
    ensemble.fit(X, y)
    joblib.dump(ensemble, ensemble_model_path)
    
    y_pred = ensemble.predict(X)
    logging.info(f"Accuracy of ensemble model: {accuracy_score(y, y_pred)}")
    logging.info(f"Classification Report for ensemble model:\n{classification_report(y, y_pred)}")