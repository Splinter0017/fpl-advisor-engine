import requests
import pandas as pd
from pathlib import Path
import logging

# --- CONFIGURATION ---
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def fetch_season_data(season):
    """
    Downloads the merged gameweek data for a specific season.
    """
    # The URL pattern for the 'Vaastav' repo
    url = f"{BASE_URL}/{season}/gws/merged_gw.csv"
    save_path = DATA_DIR / f"{season}_merged_gw.csv"
    
    logger.info(f"Downloading {season} data...")
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Error check
        
        # Save to file
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        # Quick verify
        df = pd.read_csv(save_path)
        logger.info(f"Saved {season}: {len(df)} rows (Players x Gameweeks)")
        
    except Exception as e:
        logger.error(f"Failed to download {season}: {e}")

if __name__ == "__main__":
    # Ensure folder exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("--- STARTING HISTORY INGESTION ---")
    for season in SEASONS:
        fetch_season_data(season)
    logger.info("--- COMPLETE ---")