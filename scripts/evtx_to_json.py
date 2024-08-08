import os
import json
import logging
from tqdm import tqdm
from Evtx.Evtx import Evtx
import xmltodict
import signal

# Set up logging
log_dir = '/Users/jackweekly/Desktop/Caiber/logs'
log_file = os.path.join(log_dir, 'evtx_to_json.log')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=log_file, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# Register the signal function handler
signal.signal(signal.SIGALRM, timeout_handler)

def convert_evtx_to_json(evtx_file, json_file):
    logging.debug(f"Converting {evtx_file} to {json_file}")
    try:
        with Evtx(evtx_file) as log:
            records = []
            for i, record in enumerate(log.records()):
                if i >= 1000:  # Limit the number of records processed for each file
                    break
                record_xml = record.xml()
                record_dict = xmltodict.parse(record_xml)
                records.append(record_dict)
            with open(json_file, 'w') as f:
                json.dump(records, f)
        logging.debug(f"Successfully converted {evtx_file}")
    except TimeoutException:
        logging.error(f"Timeout processing {evtx_file}")
    except Exception as e:
        logging.error(f"Error processing {evtx_file}: {e}")

def process_file(evtx_file, input_folder, output_folder):
    json_file = os.path.join(output_folder, os.path.relpath(evtx_file, input_folder) + ".json")
    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    logging.info(f"Starting processing of {evtx_file}")
    try:
        signal.alarm(60)  # Set a 60-second alarm for each file
        convert_evtx_to_json(evtx_file, json_file)
        signal.alarm(0)  # Disable the alarm
        logging.info(f"Finished processing {evtx_file}")
    except TimeoutException:
        logging.error(f"Timeout processing {evtx_file}")
    except Exception as e:
        logging.error(f"Error processing {evtx_file}: {e}")

def process_evtx_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    evtx_files = [os.path.join(root, file) 
                  for root, _, files in os.walk(input_folder) 
                  for file in files if file.endswith(".evtx")]

    logging.info(f"Found {len(evtx_files)} .evtx files to process.")

    if len(evtx_files) == 0:
        logging.error("No .evtx files found in the input directory.")
        return

    for evtx_file in tqdm(evtx_files, desc="Processing EVTX files"):
        process_file(evtx_file, input_folder, output_folder)

    logging.info("Finished processing all files.")

if __name__ == "__main__":
    input_folder = "/Users/jackweekly/Desktop/Caiber/data/raw_event_logs/EVTX-ATTACK-SAMPLES"
    output_folder = "/Users/jackweekly/Desktop/Caiber/data/processed_event_logs"
    logging.info(f"Input folder: {input_folder}")
    logging.info(f"Output folder: {output_folder}")
    process_evtx_files(input_folder, output_folder)