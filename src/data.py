import requests
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# 1. Official API (For current season)
API_BASE = "https://fantasy.premierleague.com/api"

# 2. Archive Repo (For history)
REPO_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
TRAINING_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"] 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FPLDataEngine:
    def __init__(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def fetch_archived_season(self, season):
        """
        Fetches historical data from the community archive (Vaastav Repo).
        Includes cleaning logic to fix corrupt CSV lines.
        """
        logger.info(f"[ARCHIVE] Fetching {season}...")
        
        # A. Fetch Player Stats (Merged GWs)
        url = f"{REPO_BASE}/{season}/gws/merged_gw.csv"
        path = RAW_DIR / f"{season}_merged_gw.csv"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # Save raw
            with open(path, 'wb') as f:
                f.write(response.content)
            
            # Sanitize (Fix the "Expected 42 fields, saw 49" error)
            try:
                df = pd.read_csv(path, on_bad_lines='skip')
            except TypeError:
                df = pd.read_csv(path, error_bad_lines=False) # Old pandas fallback
            
            df['season'] = season
            df.to_csv(path, index=False) # Save clean version
            logger.info(f"Stats: {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Failed to fetch stats for {season}: {e}")
            return None

        # B. Fetch Team Map (To fix the ID vs Name bug)
        url_teams = f"{REPO_BASE}/{season}/teams.csv"
        path_teams = RAW_DIR / f"{season}_teams.csv"
        try:
            r = self.session.get(url_teams)
            r.raise_for_status()
            with open(path_teams, 'wb') as f:
                f.write(r.content)
            logger.info("Teams Map acquired")
        except Exception:
            logger.warning(f"Could not fetch team map for {season}")

    def fetch_live_season(self):
        """
        Fetches the CURRENT season directly from the Official API.
        This builds a 'merged_gw.csv' equivalent from live data.
        """
        logger.info("📡 [LIVE API] Fetching Current Season...")
        
        try:
            # 1. Bootstrap: Get all players and teams
            static = self.session.get(f"{API_BASE}/bootstrap-static/").json()
            players = pd.DataFrame(static['elements'])
            teams = pd.DataFrame(static['teams'])
            
            # Save Reference Data
            current_season_label = "2025-26" 
            teams.to_csv(RAW_DIR / f"{current_season_label}_teams.csv", index=False)
            
            # 2. Get Gameweek History for EVERY player
            # We must loop through every player ID to get their match history
            all_history = []
            player_ids = players['id'].tolist()
            
            logger.info(f"Downloading history for {len(player_ids)} players (this takes ~30s)...")

            count = 0
            for pid in player_ids:
                try:
                    p_summary = self.session.get(f"{API_BASE}/element-summary/{pid}/").json()
                    history = p_summary['history'] # Past fixtures this season
                    
                    if history:
                        p_df = pd.DataFrame(history)
                        p_df['element'] = pid # Add Player ID
                        # Add name for easier debugging
                        p_df['name'] = players.loc[players['id'] == pid, 'web_name'].values[0]
                        all_history.append(p_df)
                except Exception:
                    pass
                
                count += 1
                if count % 100 == 0:
                    print(f"      ... {count}/{len(player_ids)}")
            
            if not all_history:
                logger.warning(" No match history found.")
                return

            # 3. Standardization
            live_df = pd.concat(all_history, ignore_index=True)
            live_df['season'] = current_season_label
            
            # Rename API columns to match Repo columns (Normalization)
            # API: 'total_points', 'minutes', 'opponent_team', 'was_home' (bool)
            # Repo: 'total_points', 'minutes', 'opponent_team', 'was_home' (True/False or 1/0)
            # Map position from bootstrap data
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            player_metadata = players[['id', 'element_type', 'team', 'web_name']].copy()
            player_metadata['position'] = player_metadata['element_type'].map(position_map)

            # Merge position and team into match history
            live_df = live_df.merge(
               player_metadata[['id', 'position', 'team']], 
               left_on='element', 
                right_on='id', 
                  how='left'
                )

            # Map team IDs to team names
            team_map = pd.Series(teams.name.values, index=teams.id).to_dict()
            live_df['team'] = live_df['team'].map(team_map)
            
            # Save
            path = RAW_DIR / f"{current_season_label}_merged_gw.csv"
            live_df.to_csv(path, index=False)
            logger.info(f"Live Data Built: {len(live_df)} rows")
            
        except Exception as e:
            logger.error(f"Live Fetch Failed: {e}")

    def run(self):
        logger.info("--- 1. STARTING HYBRID INGESTION ---")
        
        # 1. Archive (Training Data)
        for season in TRAINING_SEASONS:
            self.fetch_archived_season(season)
            time.sleep(1)
            
        # 2. Live (Test Data)
        self.fetch_live_season()
        
        logger.info("--- INGESTION COMPLETE ---")

if __name__ == "__main__":
    engine = FPLDataEngine()
    engine.run()