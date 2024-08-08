import os
import json
import pandas as pd
from tqdm import tqdm
import logging

# Set up logging
log_dir = '/Users/jackweekly/Desktop/Caiber/logs'
log_file = os.path.join(log_dir, 'feature_engineering.log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if os.path.exists(log_file):
    os.remove(log_file)
logging.basicConfig(filename=log_file, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def extract_features(json_file):
    logging.info(f"Extracting features from {json_file}")
    try:
        with open(json_file, 'r') as f:
            records = json.load(f)
    except Exception as e:
        logging.error(f"Error reading JSON file {json_file}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if JSON reading fails
    
    features = []
    for record in records:
        event = record.get('Event', {})
        event_data = event.get('EventData', {})
        system_data = event.get('System', {})
        
        feature = {}
        for data in event_data.get('Data', []):
            feature[data.get('@Name')] = data.get('#text', '')
        
        feature['EventID'] = system_data.get('EventID', {}).get('#text', '')
        feature['Channel'] = system_data.get('Channel', '')
        feature['Computer'] = system_data.get('Computer', '')
        feature['Provider'] = system_data.get('Provider', {}).get('@Name', '')
        feature['TimeCreated'] = system_data.get('TimeCreated', {}).get('@SystemTime', '')
        
        features.append(feature)
    
    return pd.DataFrame(features)

def process_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    json_files = [os.path.join(root, file) 
                  for root, _, files in os.walk(input_folder) 
                  for file in files if file.endswith(".json")]

    logging.info(f"Found {len(json_files)} JSON files to process in {input_folder}")

    if len(json_files) == 0:
        logging.error("No JSON files found in the input directory.")
        return
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            logging.debug(f"Processing file: {json_file}")
            df = extract_features(json_file)
            if df.empty:
                logging.error(f"Skipping file due to extraction failure: {json_file}")
                continue
            
            output_file = os.path.join(output_folder, os.path.relpath(json_file, input_folder)).replace('.json', '.csv')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            logging.info(f"Successfully processed {json_file}")
        except Exception as e:
            logging.error(f"Error processing {json_file}: {e}")

if __name__ == "__main__":
    input_folder = "/Users/jackweekly/Desktop/Caiber/data/normalized_json_logs"
    output_folder = "/Users/jackweekly/Desktop/Caiber/data/feature_engineered"
    logging.info(f"Input folder: {input_folder}")
    logging.info(f"Output folder: {output_folder}")
    process_files(input_folder, output_folder)