import os
import json
import logging
from tqdm import tqdm
from datetime import datetime

# Set up logging
log_dir = '/Users/jackweekly/Desktop/Caiber/logs'
log_file = os.path.join(log_dir, 'normalize_json.log')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_timestamp(event):
    try:
        timestamp = event['Event']['System']['TimeCreated']['@SystemTime']
        if timestamp:
            event['Event']['System']['TimeCreated']['@SystemTime'] = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).isoformat()
    except KeyError as e:
        logging.error(f"Missing key during timestamp normalization: {e}")
    except Exception as e:
        logging.error(f"Error normalizing timestamp: {e}")
    return event

def normalize_json_file(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            records = json.load(f)
            normalized_records = [normalize_timestamp(record) for record in records]
        with open(output_file, 'w') as f:
            json.dump(normalized_records, f)
    except Exception as e:
        logging.error(f"Error normalizing {input_file}: {e}")

def process_json_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    json_files = [os.path.join(root, file) 
                  for root, _, files in os.walk(input_folder) 
                  for file in files if file.endswith(".json")]
    
    for json_file in tqdm(json_files, desc="Normalizing JSON files"):
        output_file = os.path.join(output_folder, os.path.relpath(json_file, input_folder))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        logging.info(f"Starting normalization of {json_file}")
        try:
            normalize_json_file(json_file, output_file)
            logging.info(f"Finished normalization of {json_file}")
        except Exception as e:
            logging.error(f"Error normalizing {json_file}: {e}")

if __name__ == "__main__":
    input_folder = "/Users/jackweekly/Desktop/Caiber/data/processed_event_logs"
    output_folder = "/Users/jackweekly/Desktop/Caiber/data/normalized_json_logs"
    process_json_files(input_folder, output_folder)