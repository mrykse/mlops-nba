import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd

# Paths
RAW_DATA_DIR = Path('../../data/raw')
PRE_CURATED_DATA_DIR = Path('../../data/pre_curated')
CURATED_DATA_DIR = Path('../../data/curated')
os.makedirs(PRE_CURATED_DATA_DIR, exist_ok=True)
os.makedirs(CURATED_DATA_DIR, exist_ok=True)

# 'eFG%', "Rk" --> not needed
mapping = {
    "TEAM_ABBREVIATION": "Tm",
    "PLAYER_NAME": "Player",
    "POSITION": "Pos",
    "AGE": "Age",
    "GAME_ID": "G",
    "START_POSITION": "GS",
    "mins": "MP",
    "FGM": "FG",
    "FGA": "FGA",
    "FG_PCT": "FG%",
    "FG3M": "3P",
    "FG3A": "3PA",
    "FG3_PCT": "3P%",
    "2PM": "2P",
    "2PA": "2PA",
    "2P_PCT": "2P%",
    "FTM": "FT",
    "FTA": "FTA",
    "FT_PCT": "FT%",
    "OREB": "ORB",
    "DREB": "DRB",
    "REB": "TRB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "TO": "TOV",
    "PF": "PF",
    "PTS": "PTS",
}


def transformation(dataframe: pd.DataFrame, convert_duration_to_number=None) -> pd.DataFrame:
    """Transform input dataframe and add some columns."""
    dataframe["2PM"] = dataframe["FGM"] - dataframe["FG3M"]
    dataframe["2PA"] = dataframe["FGA"] - dataframe["FG3A"]
    dataframe["2P_PCT"] = dataframe["2PM"] / dataframe["2PA"]
    dataframe["mins"] = dataframe["MIN"].fillna("0:00").apply(convert_duration_to_number)
    dataframe["AGE"] = dataframe["AGE"].fillna(0).astype(int)
    return dataframe


def preprocess_csv(file_path: Path):
    # Output results
    file_name = file_path.stem  # Use the name without extension

    players = pd.read_csv(file_path, encoding='ISO-8859-1')
    now = pd.to_datetime("now").strftime("%Y%m%d%H%M%S")
    players["EFF"] = players.PTS + players.TRB + players.AST + players.STL + players.BLK - (
            players.FGA - players.FG) - (players.FTA - players.FT) - players.TOV

    # Save the preprocessed data to the specified file_path
    processed_file_path = PRE_CURATED_DATA_DIR / f"{now}-{file_name}-pre_curated_data.csv"
    players.to_csv(processed_file_path, index=False)
    print(f"Processed {file_name} : OK")

    # Append the preprocessed data to the curated parquet file
    curated_parquet_path = CURATED_DATA_DIR / 'curated_data.parquet'
    if not curated_parquet_path.exists():
        # If the parquet file doesn't exist, create it
        players.to_parquet(curated_parquet_path, index=False)
    else:
        # If the parquet file exists, append the new data
        curated_data = pd.read_parquet(curated_parquet_path)
        curated_data = pd.concat([curated_data, players], ignore_index=True)
        curated_data.to_parquet(curated_parquet_path, index=False)

        print(f"Appended {file_name} to curated_data.parquet")


class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        print(f"New file detected in the path data/raw: {event.src_path}")
        preprocess_csv(Path(event.src_path))


def monitor_directory():
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=str(RAW_DATA_DIR), recursive=False)
    observer.start()

    try:
        last_message_time = time.time()
        while True:
            current_time = time.time()
            if current_time - last_message_time >= 10:
                print("You can add a file to process any time you want.")
                last_message_time = current_time
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    monitor_directory()
