import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from mlops_nba.pre_processing.pre_processing import preprocess_csv
from mlops_nba.training.train_model import train_and_test_models

# Paths
RAW_DATA_DIR = Path('../../data/raw')
PRE_CURATED_DATA_DIR = Path('../../data/pre_curated')
CURATED_DATA_DIR = Path('../../data/curated')
os.makedirs(PRE_CURATED_DATA_DIR, exist_ok=True)
os.makedirs(CURATED_DATA_DIR, exist_ok=True)


class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        print(f"New file detected in the path data/raw: {event.src_path}")
        time.sleep(2)  # Wait for the file to be written
        preprocess_csv(Path(event.src_path))


def monitor_directory():
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=str(RAW_DATA_DIR), recursive=False)
    observer.start()

    try:
        last_message_time = time.time()  # Initialize last_message_time
        # Process existing files when the script starts
        for file_path in RAW_DATA_DIR.glob("*"):
            if file_path.is_file():
                print(f"Processing existing file: {file_path}")
                preprocess_csv(file_path)

        while True:
            current_time = time.time()
            if current_time - last_message_time >= 30:
                # You can add any additional logic here before triggering preprocessing
                preprocess_csv(
                    CURATED_DATA_DIR / 'curated_data.parquet')  # Trigger the preprocessing for curated_data.parquet
                last_message_time = current_time

                # After processing all new files, trigger the training and testing
                train_and_test_models('curated_data.parquet')

            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping the monitoring script...")
    observer.join()


if __name__ == "__main__":
    monitor_directory()
