import pandas as pd
import xgboost as xgb
import requests
import gspread
import logging
from pathlib import Path
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "src" / "fpl_model.json"
DATA_PATH = BASE_DIR / "data" / "processed" / "training_data_advanced.csv"
KEY_FILE = BASE_DIR / "config" / "service_account.json"
SHEET_NAME = "FPL_Predictions"  # <--- NEW SHEET NAME

API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model

def fetch_next_gameweek_data():
    logger.info("1. Fetching Live Data...")
    bootstrap = requests.get(API_URL).json()
    players = pd.DataFrame(bootstrap['elements'])
    teams = pd.DataFrame(bootstrap['teams'])
    
    next_event = next(e for e in bootstrap['events'] if e['is_next'])
    gw_id = next_event['id']
    logger.info(f"   -> Predicting for Gameweek {gw_id}")
    
    all_fixtures = requests.get(FIXTURES_URL).json()
    gw_fixtures = [f for f in all_fixtures if f['event'] == gw_id]
    
    fixture_map = {}
    for f in gw_fixtures:
        home = f['team_h']
        away = f['team_a']
        if home not in fixture_map: fixture_map[home] = []
        fixture_map[home].append({'opp': away, 'was_home': True})
        if away not in fixture_map: fixture_map[away] = []
        fixture_map[away].append({'opp': home, 'was_home': False})
        
    return players, teams, fixture_map, gw_id

def prepare_features(players, teams, fixture_map, historical_df):
    logger.info("2. Engineering Features...")
    rows = []
    team_name_map = pd.Series(teams.name.values, index=teams.id).to_dict()
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    # Pre-calculate Opponent Vulnerability Lookup to speed up loop
    # Group by Opponent and Position, get the latest value
    vuln_lookup = historical_df.sort_values('kickoff_time').groupby(['opponent_team', 'position']).tail(1)
    vuln_map = vuln_lookup.set_index(['opponent_team', 'position'])['opp_def_strength_vs_pos'].to_dict()

    for _, p in players.iterrows():
        team_id = p['team']
        if team_id not in fixture_map: continue
            
        matches = fixture_map[team_id]
        for m in matches:
            opp_name = team_name_map[m['opp']]
            pos_str = pos_map[p['element_type']]
            
            row = {
                'name': p['web_name'],
                'team': team_name_map[team_id],
                'position': pos_str,
                'now_cost': p['now_cost'] / 10.0,
                'was_home': str(m['was_home']),
                'opponent_team': opp_name,
                # Lags (Proxies from API)
                'mean_pts_3': float(p['form']),
                'mean_mins_3': 90.0 if p['status'] == 'a' else 0.0, # Optimistic minutes
                'mean_threat_3': float(p['ict_index']) * 10,
                'mean_creativity_3': float(p['creativity']) / 3,
                'last_3_xp_delta': 0.0,
                'team_strength_diff': 0.0 # Simplification for V1
            }
            
            # Lookup Vulnerability
            row['opp_def_strength_vs_pos'] = vuln_map.get((opp_name, pos_str), 2.5) # Default 2.5 if new team
            
            rows.append(row)
            
    return pd.DataFrame(rows)

def upload_to_cloud(df):
    logger.info(f"4. Uploading to Google Sheets ({SHEET_NAME})...")
    try:
        gc = gspread.service_account(filename=str(KEY_FILE))
        sh = gc.open(SHEET_NAME)
        worksheet = sh.get_worksheet(0)
        worksheet.clear()
        
        # --- THE FIX ---
        # Create a copy so we don't break the original dataframe
        upload_df = df.copy()
        
        # Convert all "Category" columns back to simple "Strings"
        # This allows us to put '0' or anything else in them without crashing
        for col in upload_df.select_dtypes(['category']):
            upload_df[col] = upload_df[col].astype(str)
            
        # Sanitize for JSON (NaN -> 0)
        upload_df = upload_df.fillna(0)
        
        # Upload
        payload = [upload_df.columns.values.tolist()] + upload_df.values.tolist()
        worksheet.update(range_name='A1', values=payload)
        logger.info("✅ Cloud Sync Complete.")
    except Exception as e:
        logger.error(f"❌ Cloud Upload Failed: {e}")
        # Print the full error for debugging
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    hist_df = pd.read_csv(DATA_PATH)
    players, teams, fixture_map, gw = fetch_next_gameweek_data()
    
    prediction_df = prepare_features(players, teams, fixture_map, hist_df)
    
    cat_cols = ['position', 'opponent_team', 'was_home']
    for col in cat_cols:
        prediction_df[col] = prediction_df[col].astype('category')
        
    logger.info("3. Predicting Points...")
    model = load_model()
    
    # Feature order must match training exactly
    features = [
        'now_cost', 'opp_def_strength_vs_pos', 'last_3_xp_delta', 'team_strength_diff',
        'mean_pts_3', 'mean_threat_3', 'mean_creativity_3', 'mean_mins_3',
        'was_home', 'position', 'opponent_team'
    ]
    
    preds = model.predict(prediction_df[features])
    prediction_df['predicted_points'] = preds
    
    # Final Report
    report = prediction_df[['name', 'position', 'team', 'opponent_team', 'now_cost', 'predicted_points']]
    report = report.sort_values(by='predicted_points', ascending=False)
    
    # 1. Save Local CSV (Backup)
    local_path = BASE_DIR / "data" / f"predictions_gw{gw}.csv"
    report.to_csv(local_path, index=False)
    
    # 2. Push to Cloud (For Gemini)
    upload_to_cloud(report)