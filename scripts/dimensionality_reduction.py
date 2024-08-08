import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import logging
import logging.handlers
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Process

# Set up logging
log_dir = '/Users/jackweekly/Desktop/Caiber/logs'
log_file = os.path.join(log_dir, 'dimensionality_reduction.log')

def configure_logging(log_queue):
    handler = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    handler.setLevel(logging.DEBUG)
    
def listener_configurer():
    root = logging.getLogger()
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

def listener_process(queue):
    listener_configurer()
    listener = logging.handlers.QueueListener(queue, *logging.getLogger().handlers)
    listener.start()
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception as e:
            print(f"Listener failed to handle record: {e}")

def preprocess_features(df, col_threshold=0.2, row_threshold=0.2):
    essential_cols = ['EventID', 'Channel', 'Computer', 'Provider', 'TimeCreated']
    features = df.drop(columns=essential_cols)
    logging.debug(f"Features selected with shape: {features.shape}")

    logging.debug(f"NaN summary before processing:\n{features.isnull().sum()}")
    features = features.loc[:, features.isnull().mean() < col_threshold]
    logging.debug(f"Features after dropping columns with >{col_threshold*100}% NaN values: {features.shape}")

    features = features.dropna(thresh=int(row_threshold * features.shape[1]))
    logging.debug(f"Features after dropping rows with >{row_threshold*100}% NaN values: {features.shape}")

    features = features.apply(pd.to_numeric, errors='coerce')
    logging.debug(f"Converted features to numeric with shape: {features.shape}")

    features = features.select_dtypes(include=[np.number])
    logging.debug(f"Features after retaining only numeric columns: {features.shape}")

    if features.isnull().values.any():
        logging.warning(f"NaN values found in features. Filling NaNs with column mean.")
        features = features.fillna(features.mean())

    features = features.dropna(axis=1)
    logging.debug(f"Features after dropping columns with NaN values post-filling: {features.shape}")

    if features.isnull().values.any():
        logging.error(f"NaN values still present after processing. Dropping rows with NaN values.")
        features.dropna(inplace=True)

    return features

def apply_pca(input_file, output_file, n_components=2):
    logging.info(f"Applying PCA to {input_file}")
    start_time = time.time()
    initial_columns, initial_rows, final_columns, final_rows = 0, 0, 0, 0
    try:
        df = pd.read_csv(input_file)
        initial_columns = df.shape[1]
        initial_rows = df.shape[0]
        logging.debug(f"Loaded CSV with shape: {df.shape}")

        features = preprocess_features(df)
        final_columns = features.shape[1]
        final_rows = features.shape[0]

        if final_columns < n_components:
            logging.error(f"Not enough features to apply PCA in {input_file}. Skipping file.")
            return initial_columns, initial_rows, 0, 0

        variance = features.var()
        if (variance == 0).any():
            logging.error(f"Zero variance detected in features of {input_file}. Skipping file.")
            return initial_columns, initial_rows, 0, 0

        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(features)
        logging.debug(f"Explained variance by each component: {pca.explained_variance_ratio_}")

        pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
        result_df = pd.concat([df[['EventID', 'Channel', 'Computer', 'Provider', 'TimeCreated']], pca_df], axis=1)
        result_df.to_csv(output_file, index=False)
        logging.info(f"Successfully applied PCA and saved to {output_file}")
    except Exception as e:
        logging.error(f"Error applying PCA to {input_file}: {e}", exc_info=True)
    finally:
        end_time = time.time()
        logging.info(f"Time taken for PCA on {input_file}: {end_time - start_time} seconds")
        return initial_columns, initial_rows, final_columns, final_rows

def process_file(file_info):
    log_queue, csv_file, output_folder, n_components = file_info
    configure_logging(log_queue)
    try:
        output_file = os.path.join(output_folder, os.path.relpath(csv_file, input_folder))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        return apply_pca(csv_file, output_file, n_components)
    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}", exc_info=True)
        return 0, 0, 0, 0

def process_files(input_folder, output_folder, n_components=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = [os.path.join(root, file)
                 for root, _, files in os.walk(input_folder)
                 for file in files if file.endswith(".csv")]

    logging.info(f"Found {len(csv_files)} CSV files to process in {input_folder}")

    if len(csv_files) == 0:
        logging.error("No CSV files found in the input directory.")
        return

    file_info_list = [(log_queue, csv_file, output_folder, n_components) for csv_file in csv_files]

    successful_files = 0
    skipped_files = 0
    total_columns_dropped = 0
    total_rows_dropped = 0

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, file_info_list), total=len(file_info_list), desc="Applying PCA to CSV files"))

    for result in results:
        initial_columns, initial_rows, final_columns, final_rows = result
        if final_columns > 0 and final_rows > 0:
            successful_files += 1
            total_columns_dropped += initial_columns - final_columns
            total_rows_dropped += initial_rows - final_rows
        else:
            skipped_files += 1

    logging.info(f"Successfully processed {successful_files} files.")
    logging.info(f"Skipped {skipped_files} files due to insufficient features or errors.")
    logging.info(f"Total columns dropped: {total_columns_dropped}")
    logging.info(f"Total rows dropped: {total_rows_dropped}")

if __name__ == "__main__":
    input_folder = "/Users/jackweekly/Desktop/Caiber/data/feature_engineered"
    output_folder = "/Users/jackweekly/Desktop/Caiber/data/reduced_dimension"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.exists(log_file):
        os.remove(log_file)
    
    manager = Manager()
    log_queue = manager.Queue()
    listener = Process(target=listener_process, args=(log_queue,))
    listener.start()
    process_files(input_folder, output_folder)
    log_queue.put_nowait(None)  # Signal the listener to stop
    listener.join()