import subprocess
import logging
import os
from tqdm import tqdm
from prettytable import PrettyTable
import time
import json
import pandas as pd

# Set up logging for the master script
log_file = '/Users/jackweekly/Desktop/Caiber/logs/run_all.log'
if not os.path.exists('/Users/jackweekly/Desktop/Caiber/logs'):
    os.makedirs('/Users/jackweekly/Desktop/Caiber/logs')
if os.path.exists(log_file):
    os.remove(log_file)
logging.basicConfig(filename=log_file, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(script_path, log_path):
    logging.info(f"Running script: {script_path}")
    pbar = tqdm(total=100, desc=f"Running {os.path.basename(script_path)}", position=1, leave=False)
    
    try:
        process = subprocess.run(['python3', script_path], capture_output=True, text=True, timeout=600)
        pbar.update(100)
        pbar.close()
        
        if process.returncode == 0:
            logging.info(f"Successfully ran {script_path}")
            return (script_path, "Success", "")
        else:
            logging.error(f"Error running {script_path}")
            logging.error(process.stderr)
            return (script_path, "Failed", process.stderr)
    except subprocess.TimeoutExpired:
        pbar.close()
        logging.error(f"Timeout expired running {script_path}")
        return (script_path, "Timeout", "Timeout expired")
    except Exception as e:
        pbar.close()
        logging.error(f"Exception running {script_path}: {e}")
        return (script_path, "Exception", str(e))

def verify_files(directory, extension, check_function):
    logging.info(f"Verifying files in {directory} with extension {extension}")
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
    for file in files[:10]:  # Check the first 10 files
        if not check_function(file):
            logging.error(f"Verification failed for file {file}")
            return False
    logging.info(f"All files in {directory} passed verification")
    return True

def check_json_format(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(record, dict) for record in data):
            return True
    except Exception as e:
        logging.error(f"Error checking JSON format for {file_path}: {e}")
    return False

def check_csv_format(file_path):
    try:
        df = pd.read_csv(file_path)
        if not df.empty:
            return True
    except Exception as e:
        logging.error(f"Error checking CSV format for {file_path}: {e}")
    return False

if __name__ == "__main__":
    scripts = [
        ('/Users/jackweekly/Desktop/Caiber/scripts/evtx_to_json.py', '/Users/jackweekly/Desktop/Caiber/logs/evtx_to_json.log', '/Users/jackweekly/Desktop/Caiber/data/processed_event_logs', '.json', check_json_format),
        ('/Users/jackweekly/Desktop/Caiber/scripts/json_normalizer.py', '/Users/jackweekly/Desktop/Caiber/logs/normalize_json.log', '/Users/jackweekly/Desktop/Caiber/data/normalized_json_logs', '.json', check_json_format),
        ('/Users/jackweekly/Desktop/Caiber/scripts/feature_engineering.py', '/Users/jackweekly/Desktop/Caiber/logs/feature_engineering.log', '/Users/jackweekly/Desktop/Caiber/data/feature_engineered', '.csv', check_csv_format),
        ('/Users/jackweekly/Desktop/Caiber/scripts/dimensionality_reduction.py', '/Users/jackweekly/Desktop/Caiber/logs/dimensionality_reduction.log', '/Users/jackweekly/Desktop/Caiber/data/reduced_dimension', '.csv', check_csv_format),
        ('/Users/jackweekly/Desktop/Caiber/scripts/model_training/train_model.py', '/Users/jackweekly/Desktop/Caiber/logs/train_model.log', '/Users/jackweekly/Desktop/Caiber/data/models', '.joblib', lambda x: True),
        ('/Users/jackweekly/Desktop/Caiber/scripts/model_training/evaluate_model.py', '/Users/jackweekly/Desktop/Caiber/logs/evaluate_model.log', '', '', lambda x: True),
        ('/Users/jackweekly/Desktop/Caiber/scripts/model_training/ensemble_model.py', '/Users/jackweekly/Desktop/Caiber/logs/ensemble_model.log', '/Users/jackweekly/Desktop/Caiber/data/models', '.joblib', lambda x: True)
    ]
    
    results = []
    
    for script, log_path, output_dir, file_ext, check_func in tqdm(scripts, desc="Running scripts", position=0):
        logging.info(f"Starting to run script: {script}")
        result = run_script(script, log_path)
        results.append(result)
        logging.info(f"Finished running script: {script} with status: {result[1]}")
        
        if result[1] == "Success" and output_dir:
            if verify_files(output_dir, file_ext, check_func):
                logging.info(f"Verified output files for {script} in {output_dir}")
            else:
                logging.error(f"Verification failed for output files in {output_dir}")
                results[-1] = (script, "Verification Failed", f"Output files in {output_dir} did not pass verification")
    
    # Create a summary table
    table = PrettyTable()
    table.field_names = ["Script", "Status", "Details"]
    
    for script, status, details in results:
        table.add_row([os.path.basename(script), status, details if status != "Success" else ""])
    
    print(table)
    logging.info("\n" + str(table))