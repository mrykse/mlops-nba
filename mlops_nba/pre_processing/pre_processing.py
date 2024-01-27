import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
from mlops_nba.training.train_model import train_and_test_models
import hashlib

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
    "FT": "FT",
    "FTA": "FTA",
    "FT_PCT": "FT%",
    "OREB": "ORB",
    "DREB": "DRB",
    "TRB": "TRB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "TOV": "TOV",
    "PF": "PF",
    "PTS": "PTS",
}


def transformation(dataframe: pd.DataFrame, convert_duration_to_number=None) -> pd.DataFrame:
    """Transform input dataframe and add some columns."""
    dataframe["2PM"] = dataframe[mapping["FGM"]] - dataframe[mapping["FG3M"]]
    dataframe["2PA"] = dataframe[mapping["FGA"]] - dataframe[mapping["FG3A"]]
    dataframe["2P_PCT"] = dataframe["2PM"] / dataframe["2PA"]
    dataframe["AGE"] = dataframe[mapping["AGE"]].fillna(0).astype(int)

    # Add efficiency column using direct column names
    dataframe["efficiency"] = (
            dataframe["PTS"]
            + dataframe["TRB"]
            + dataframe["AST"]
            + dataframe["STL"]
            + dataframe["BLK"]
            - (dataframe["FGA"] - dataframe["FG"])
            - (dataframe["FTA"] - dataframe["FT"])
            - dataframe["TOV"]
    )

    # Add rising_stars column based on specified conditions
    AGE_THRESHOLD = 23
    POINTS_THRESHOLD = 10
    EFFICIENCY_THRESHOLD = 12
    dataframe["rising_stars"] = (
            (dataframe["efficiency"] >= EFFICIENCY_THRESHOLD)
            & (dataframe["PTS"] >= POINTS_THRESHOLD)
            & (dataframe["AGE"] <= AGE_THRESHOLD)
    ).astype(int)

    return dataframe


# Add a set to keep track of processed file names
processed_files = set()


def preprocess_csv(file_path: Path):
    # Output results
    file_name = file_path.stem  # Use the name without extension

    # Check if the file has already been processed
    file_hash = hashlib.md5(file_name.encode('utf-8')).hexdigest()
    if file_hash in processed_files:
        print(f"File {file_name} has already been processed.")
        return

    try:
        players = pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    now = pd.to_datetime("now").strftime("%Y%m%d%H%M%S")
    players = transformation(players)  # Add efficiency column
    players["efficiency"] = players.PTS + players.TRB + players.AST + players.STL + players.BLK - (
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

    # If the parquet file exists, append the new data
    curated_data = pd.read_parquet(curated_parquet_path)
    curated_data = pd.concat([curated_data, players], ignore_index=True)
    curated_data.to_parquet(curated_parquet_path, index=False)

    print(f"Appended {file_name} to curated_data.parquet")

    # Add the processed file name to the set
    processed_files.add(file_hash)

    # Trigger the separate pipeline for training and testing the model
    train_and_test_models('curated_data.parquet')


