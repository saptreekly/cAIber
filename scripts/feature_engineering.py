import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def extract_features(json_file):
    with open(json_file, 'r') as f:
        records = json.load(f)
    
    features = []
    for record in records:
        event = record['Event']
        event_data = event['EventData']
        system_data = event['System']
        
        feature = {}
        for data in event_data['Data']:
            feature[data['@Name']] = data.get('#text', '')
        
        feature['EventID'] = system_data['EventID']['#text']
        feature['Channel'] = system_data['Channel']
        feature['Computer'] = system_data['Computer']
        feature['Provider'] = system_data['Provider']['@Name']
        feature['TimeCreated'] = system_data['TimeCreated']['@SystemTime']
        
        features.append(feature)
    
    return pd.DataFrame(features)

def process_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    json_files = [os.path.join(root, file) 
                  for root, _, files in os.walk(input_folder) 
                  for file in files if file.endswith(".json")]
    
    for json_file in json_files:
        df = extract_features(json_file)
        output_file = os.path.join(output_folder, os.path.relpath(json_file, input_folder)).replace('.json', '.csv')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_folder = "../data/normalized_json_logs"
    output_folder = "../data/feature_engineered"
    process_files(input_folder, output_folder)