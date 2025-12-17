import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FPLInferenceEngine:
    """
    Forensic analysis of player potential across multiple time horizons.
    
    Mathematical Foundation:
    E[Points] = P(Play) * E[Points | Play]
    
    Architecture:
    - Feature Alignment: Ensures matrix X matches model signature M(X).
    - Multi-Horizon Aggregation: Sum(xP_t, xP_t+1, xP_t+2) -> Asset Value.
    """
    
    def __init__(self, models: Dict, df: pd.DataFrame, features: List[str]):
        self.models = models
        self.df = df.copy()
        self.features = features
        
    def _get_aligned_features(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """
        Critical Safety: Aligns input X to the specific features the model expects.
        Prevents LightGBM shape errors by forcing exact column match.
        """
        if hasattr(model, "booster_"):
            expected = model.booster_.feature_name()
        elif hasattr(model, "_Booster"):
            expected = model._Booster.feature_name()
        elif hasattr(model, "feature_name"):
            expected = model.feature_name()
        else:
            return X 
            
        # Reindex to force exact structure match (fill missing with 0)
        return X.reindex(columns=expected, fill_value=0)

    def _predict_horizon(self, horizon: int, X: pd.DataFrame, current_gw: int) -> None:
        """
        Runs inference for a specific time horizon (h).
        Math: y_hat_{t+h} = f_h(X_t)
        """
        # 1. Regressor (The Baseline Points)
        reg_model = self.models.get(f'regressor_h{horizon}')
        if reg_model:
            X_aligned = self._get_aligned_features(reg_model, X)
            if hasattr(reg_model, "predict"):
                self.df[f'xP_GW{current_gw + horizon - 1}'] = reg_model.predict(X_aligned)
            
        # 2. Classifier (The "Upside" Probability)
        clf_model = self.models.get(f'classifier_h{horizon}')
        if clf_model:
            X_aligned = self._get_aligned_features(clf_model, X)
            if hasattr(clf_model, "predict_proba"):
                # Probability of Class 1 (e.g., Return > 5 pts)
                self.df[f'prob_upside_GW{current_gw + horizon - 1}'] = clf_model.predict_proba(X_aligned)[:, 1]
    
    def generate_report(self, current_gw: int) -> pd.DataFrame:
        """
        Aggregates predictions into a transfer planner.
        """
        # Prepare base feature matrix
        X = self.df[self.features].fillna(0)
        
        # Run inference for horizons 1, 2, 3
        horizons = [1, 2, 3]
        for h in horizons:
            self._predict_horizon(h, X, current_gw)
            
        # --- AGGREGATION LOGIC ---
        xp_cols = [f'xP_GW{current_gw + h - 1}' for h in horizons]
        available_xp_cols = [col for col in xp_cols if col in self.df.columns]
        
        if available_xp_cols:
            self.df['xP_3GW_Total'] = self.df[available_xp_cols].sum(axis=1)
        else:
            self.df['xP_3GW_Total'] = 0.0
        
        # Select Metadata Columns
        base_meta = ['name', 'team', 'position', 'value', 'form_score', 'ict_index']
        optional_meta = ['minutes', 'opponent_team', 'selected', 'web_name', 'id']
        meta_cols = base_meta + [c for c in optional_meta if c in self.df.columns]
        
        output_cols = meta_cols + available_xp_cols + ['xP_3GW_Total']
        
        # Filter: only active players to reduce noise
        if 'minutes' in self.df.columns:
            active_players = self.df[self.df['minutes'] > 0].copy()
        else:
            active_players = self.df.copy()
            
        return active_players[output_cols].sort_values('xP_3GW_Total', ascending=False)


def train_jit_models(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Trains fresh models on the fly (Just-In-Time).
    
    Reasoning:
    Guarantees that Training Schema == Inference Schema.
    Eliminates 'feature mismatch' errors caused by loading stale .pkl files.
    """
    logger.info(f"INITIATING JIT TRAINING (Features: {len(features)})")
    models = {}
    
    # Sort for correct time-shifting
    df_train = df.sort_values(['element', 'season', 'GW']).copy()
    
    for h in [1, 2, 3]:
        target_col = f'target_h{h}'
        # Shift: The points I score in h weeks are my target today
        df_train[target_col] = df_train.groupby('element')['total_points'].shift(-h)
        
        # Drop nulls (last gameweeks of history have no future target)
        valid_mask = df_train[target_col].notna()
        
        X = df_train.loc[valid_mask, features]
        y = df_train.loc[valid_mask, target_col]
        
        # LightGBM Regressor (Optimized for speed/accuracy balance)
        model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        model.fit(X, y)
        models[f'regressor_h{h}'] = model
        logger.info(f"  [OK] Trained regressor_h{h} on {len(X):,} samples")
        
    return models


def get_model_features(df: pd.DataFrame) -> List[str]:
    """Identifies feature columns, excluding metadata/targets."""
    exclude_patterns = [
        'element', 'name', 'position', 'team', 'season', 'GW',
        'total_points', 'fixture', 'kickoff_time', 'opponent_team',
        'match_score', 'value', 'selected', 'transfers_', 'was_home',
        'team_h_score', 'team_a_score', 'target_', 'xP_', 'prob_'
    ]
    
    feature_cols = []
    for col in df.columns:
        if any(pattern in col for pattern in exclude_patterns):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
            
    return feature_cols