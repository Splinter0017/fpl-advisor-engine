import requests
import pandas as pd
import gspread
import logging
from pathlib import Path

# --- CONFIGURATION & PATHS ---
# This ensures code works regardless of where you run it from
BASE_DIR = Path(__file__).resolve().parent.parent
KEY_FILE = BASE_DIR / "config" / "service_account.json"
SHEET_NAME = "FPL_data"
API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_fpl_data():
    """
    EXTRACT: Grabs the raw JSON database from the FPL API.
    """
    logger.info("Connecting to FPL API...")
    try:
        response = requests.get(API_URL)
        response.raise_for_status() # Raise error if website is down
        data = response.json()
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return None, None

    # We need both 'elements' (players) and 'teams' (for names)
    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    
    return players_df, teams_df

def transform_data(players_df, teams_df):
    """
    TRANSFORM: Cleans data and calculates 'Smart Metrics'.
    """
    logger.info("Processing data and calculating metrics...")
    
    # 1. Map Team IDs to Names (e.g., 1 -> 'Arsenal')
    team_map = pd.Series(teams_df.name.values, index=teams_df.id).to_dict()
    players_df['team_name'] = players_df['team'].map(team_map)
    
    # 2. Map Position IDs (1=GK, 2=DEF, 3=MID, 4=FWD)
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    players_df['position'] = players_df['element_type'].map(pos_map)
    
    # 3. Clean Types
    players_df['now_cost'] = players_df['now_cost'] / 10.0 # Convert 120 -> 12.0
    players_df['form'] = pd.to_numeric(players_df['form'])
    players_df['ict_index'] = pd.to_numeric(players_df['ict_index'])
    
    # 4. THE MATH: ROI Metric (Return On Investment)
    # Logic: Who is giving me the most form per million spent?
    # Equation: Form / Cost
    players_df['roi_index'] = players_df.apply(
        lambda x: round((x['form'] / x['now_cost']), 2) if x['now_cost'] > 0 else 0, axis=1
    )

    # 5. Filter & Sort
    # We only care about active players
    cols = ['web_name', 'position', 'team_name', 'now_cost', 'form', 'roi_index', 'total_points', 'ict_index']
    final_df = players_df[cols].sort_values(by='form', ascending=False)
    
    return final_df

def upload_to_sheets(df):
    """
    LOAD: Pushes the clean data to the Google Sheet for the Gemini Gem.
    """
    logger.info(f"Uploading {len(df)} rows to Google Sheets...")
    
    try:
        # Connect using the key in /config/
        gc = gspread.service_account(filename=str(KEY_FILE))
        sh = gc.open(SHEET_NAME)
        worksheet = sh.get_worksheet(0) # First tab
        
        # Clear old data
        worksheet.clear()
        
        # Format for upload: Header + Rows
        payload = [df.columns.values.tolist()] + df.values.tolist()
        worksheet.update(payload)
        
        logger.info("✅ Success! FPL_Brain is updated.")
        
    except FileNotFoundError:
        logger.error(f"❌ Could not find key file at: {KEY_FILE}")
        logger.error("Did you move service_account.json to the config folder?")
    except Exception as e:
        logger.error(f"❌ Google Sheets Error: {e}")

if __name__ == "__main__":
    # The Pipeline Execution
    raw_players, raw_teams = fetch_fpl_data()
    
    if raw_players is not None:
        clean_data = transform_data(raw_players, raw_teams)
        upload_to_sheets(clean_data)