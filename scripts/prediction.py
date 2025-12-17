import sys
import pandas as pd
from pathlib import Path


FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
sys.path.append(str(PROJECT_ROOT))  

from src.inference import FPLInferenceEngine, train_jit_models, get_model_features

# Configuration
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

TARGET_SEASON = "2025-26"  # Current Campaign

def main():
    print("--- FPL ADVISOR ENGINE: GW PREDICTION ---")
    print(f"Root: {PROJECT_ROOT}")
    
    # 1. LOAD DATA
    data_path = PROCESSED_DIR / "fpl_features_engineered.csv"
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        print("Run 'src/feature_engineering.py' first.")
        return

    df_full = pd.read_csv(data_path)
    print(f"Loaded Data: {len(df_full):,} rows")

    # 2. IDENTIFY CONTEXT
    # We predict for the "Next" Gameweek relative to the latest data
    df_season = df_full[df_full['season'] == TARGET_SEASON].copy()
    if df_season.empty:
        print(f"ERROR: No data for season {TARGET_SEASON}")
        return

    latest_gw = df_season['GW'].max()
    prediction_gw = latest_gw + 1
    
    print(f"Latest Data: GW {latest_gw}")
    print(f"Target Prediction: GW {prediction_gw}")

    # 3. PREPARE PREDICTION SET
    # We use the state of the world at 'latest_gw' to predict 'prediction_gw'
    df_predict = df_season[df_season['GW'] == latest_gw].copy()
    
    # 4. JIT MODEL TRAINING
    # Train on everything up to the prediction point
    features = get_model_features(df_full)
    models = train_jit_models(df_full, features)
    
    # 5. INFERENCE
    engine = FPLInferenceEngine(models, df_predict, features)
    planner = engine.generate_report(current_gw=prediction_gw)
    
    # 6. EXPORT
    filename = f"fpl_predictions_GW{prediction_gw}.csv"
    output_path = OUTPUTS_DIR / filename
    planner.to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] Report generated: {output_path}")
    
    # 7. TERMINAL SUMMARY
    print(f"\n--- TOP TARGETS FOR GW {prediction_gw} ---")
    cols = ['name', 'team', 'position', 'value', f'xP_GW{prediction_gw}', 'xP_3GW_Total']
    print(planner[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()