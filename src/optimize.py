import pulp
import pandas as pd
import gspread
from pathlib import Path

# --- CONFIGURATION ---
BUDGET = 100.0
# Path to your Google Cloud Key
KEY_FILE = Path(__file__).parent.parent / "config" / "service_account.json"
SHEET_NAME = "FPL_Optimization"

def solve_squad_15(df, budget=100.0):
    print(f"Optimizing Full Squad (15 Players) for Budget: £{budget}m")
    
    # Clean data
    df = df.drop_duplicates(subset=['name'])
    df = df[df['now_cost'] > 0]
    
    prob = pulp.LpProblem("FPL_Squad_15", pulp.LpMaximize)
    player_ids = df.index
    
    # Binary variable: 1 if selected, 0 if not
    x = pulp.LpVariable.dicts("player", player_ids, cat='Binary')
    
    # --- CONSTRAINTS ---
    
    # 1. Total Cost <= 100m
    prob += pulp.lpSum([df.loc[i, 'now_cost'] * x[i] for i in player_ids]) <= budget
    
    # 2. Total Players = 15
    prob += pulp.lpSum([x[i] for i in player_ids]) == 15
    
    # 3. Exact Position Counts
    prob += pulp.lpSum([x[i] for i in player_ids if df.loc[i, 'position'] == 'GK']) == 2
    prob += pulp.lpSum([x[i] for i in player_ids if df.loc[i, 'position'] == 'DEF']) == 5
    prob += pulp.lpSum([x[i] for i in player_ids if df.loc[i, 'position'] == 'MID']) == 5
    prob += pulp.lpSum([x[i] for i in player_ids if df.loc[i, 'position'] == 'FWD']) == 3
    
    # 4. Team Limit (Max 3 per team)
    teams = df['team'].unique()
    for t in teams:
        prob += pulp.lpSum([x[i] for i in player_ids if df.loc[i, 'team'] == t]) <= 3

    # --- OBJECTIVE FUNCTION ---
    # Maximize total predicted points of the whole squad
    prob += pulp.lpSum([df.loc[i, 'predicted_points'] * x[i] for i in player_ids])
    
    # Solve
    # msg=0 hides the solver logs to keep terminal clean
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        print("❌ Could not find a valid squad.")
        return None
        
    selected_indices = [i for i in player_ids if x[i].varValue == 1]
    squad = df.loc[selected_indices].copy()
    
    return squad

def pick_starting_11(squad):
    """
    Given the optimal 15, pick the best legal 11 to start.
    """
    squad = squad.sort_values(by='predicted_points', ascending=False)
    
    # Always pick best Goalkeeper
    gk = squad[squad['position'] == 'GK'].head(1)
    outfield = squad[squad['position'] != 'GK']
    
    # Pick top 10 outfielders
    best_10 = outfield.head(10)
    
    # Check formation constraints (Min 3 DEF, Min 1 FWD, Min 2 MID is standard)
    def_count = len(best_10[best_10['position'] == 'DEF'])
    mid_count = len(best_10[best_10['position'] == 'MID'])
    fwd_count = len(best_10[best_10['position'] == 'FWD'])
    
    # Force min 3 Defenders
    if def_count < 3:
        needed = 3 - def_count
        # Find defenders currently on bench
        bench_defs = squad[
            (squad['position'] == 'DEF') & 
            (~squad.index.isin(best_10.index))
        ].head(needed)
        
        # Drop worst attackers from starting XI to make room
        drop_candidates = best_10[best_10['position'] != 'DEF'].tail(needed)
        
        best_10 = pd.concat([
            best_10.drop(drop_candidates.index),
            bench_defs
        ])

    # Force min 1 Forward (rare edge case but safe to include)
    if fwd_count < 1:
        needed = 1 - fwd_count
        bench_fwds = squad[
            (squad['position'] == 'FWD') & 
            (~squad.index.isin(best_10.index))
        ].head(needed)
        
        drop_candidates = best_10[best_10['position'] != 'FWD'].tail(needed)
        best_10 = pd.concat([
            best_10.drop(drop_candidates.index),
            bench_fwds
        ])

    starting_11 = pd.concat([gk, best_10])
    bench = squad.drop(starting_11.index)
    
    return starting_11, bench

def upload_optimization(starters, bench):
    print(f"Uploading Squad to Google Sheets ({SHEET_NAME})...")
    try:
        gc = gspread.service_account(filename=str(KEY_FILE))
        
        # Open or Create the Sheet
        try:
            sh = gc.open(SHEET_NAME)
        except gspread.SpreadsheetNotFound:
            sh = gc.create(SHEET_NAME)
            # You might need to share this sheet with your personal email manually via print link
            print(f"   Note: Created new sheet '{SHEET_NAME}'. Share it with your personal email.")
            
        worksheet = sh.get_worksheet(0)
        worksheet.clear()
        
        # Prepare Data
        starters['role'] = 'Starter'
        bench['role'] = 'Bench'
        full_squad = pd.concat([starters, bench])
        
        # Sanitize for JSON (Category -> Str, NaN -> 0)
        for col in full_squad.select_dtypes(['category']):
            full_squad[col] = full_squad[col].astype(str)
        full_squad = full_squad.fillna(0)
        
        # Select clean columns
        export_cols = ['role', 'name', 'position', 'team', 'opponent_team', 'now_cost', 'predicted_points']
        final_data = full_squad[export_cols]
        
        # Upload
        payload = [final_data.columns.values.tolist()] + final_data.values.tolist()
        worksheet.update(range_name='A1', values=payload)
        print("Cloud Sync Complete.")
        
    except Exception as e:
        print(f"Cloud Upload Failed: {e}")

if __name__ == "__main__":
    # Load predictions
    data_dir = Path(__file__).parent.parent / "data"
    files = list(data_dir.glob("predictions_gw*.csv"))
    
    if not files:
        print("No prediction files found. Run predict.py first.")
        exit()
        
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading: {latest_file.name}")
    df = pd.read_csv(latest_file)
    
    # 1. Optimize
    squad_15 = solve_squad_15(df, BUDGET)
    
    if squad_15 is not None:
        # 2. Pick Starters
        starters, bench = pick_starting_11(squad_15)
        
        # 3. Print Local
        print("\n STARTING XI ")
        print(starters[['name', 'position', 'team', 'predicted_points']])
        print(f"\nTotal Expected Points: {starters['predicted_points'].sum():.2f}")
        
        # 4. Upload to Cloud
        upload_optimization(starters, bench)