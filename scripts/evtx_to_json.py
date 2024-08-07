import os
import json
import logging
from tqdm import tqdm
from Evtx.Evtx import Evtx
import xmltodict
import subprocess

# Set up logging
log_file = '../logs/evtx_to_json.log'
if os.path.exists(log_file):
    os.remove(log_file)
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def convert_evtx_to_json(evtx_file, json_file):
    try:
        with Evtx(evtx_file) as log:
            records = []
            for record in log.records():
                record_xml = record.xml()
                record_dict = xmltodict.parse(record_xml)
                records.append(record_dict)
            with open(json_file, 'w') as f:
                json.dump(records, f)
    except Exception as e:
        logging.error(f"Error processing {evtx_file}: {e}")

def process_evtx_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    evtx_files = [os.path.join(root, file) 
                  for root, _, files in os.walk(input_folder) 
                  for file in files if file.endswith(".evtx")]
    
    for evtx_file in tqdm(evtx_files, desc="Processing EVTX files"):
        json_file = os.path.join(output_folder, os.path.relpath(evtx_file, input_folder) + ".json")
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        
        logging.info(f"Starting processing of {evtx_file}")
        try:
            convert_evtx_to_json(evtx_file, json_file)
            logging.info(f"Finished processing {evtx_file}")
        except Exception as e:
            logging.error(f"Error processing {evtx_file}: {e}")

if __name__ == "__main__":
    input_folder = "../data/raw_event_logs/EVTX-ATTACK-SAMPLES"
    output_folder = "../data/preprocessed_json"
    process_evtx_files(input_folder, output_folder)
    
    # Automatically execute JSON_normalizer.py
    subprocess.run(["python", "scripts/json_normalizer.py"])