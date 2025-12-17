import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FPLInferenceEngine:
    """
    Forensic analysis of player potential across multiple time horizons.
    
    Architecture: Two-Stage Hurdle Model
    1. Classifier: P(Play) - Probability of appearance
    2. Regressor: E[Points|Play] - Expected return conditional on playing
    3. Final xP: P(Play) * E[Points|Play]
    """
    
    def __init__(self, models: Dict, df: pd.DataFrame, features: List[str]):
        self.models = models
        self.df = df.copy()
        self.features = features
        
    def _get_aligned_features(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """Aligns input X to the specific features the model expects."""
        if hasattr(model, "booster_"):
            expected = model.booster_.feature_name()
        elif hasattr(model, "_Booster"):
            expected = model._Booster.feature_name()
        elif hasattr(model, "feature_name"):
            expected = model.feature_name()
        else:
            return X 
        return X.reindex(columns=expected, fill_value=0)

    def _predict_horizon(self, horizon: int, X: pd.DataFrame, current_gw: int) -> None:
        """
        Runs inference for a specific time horizon (h).
        """
        suffix = f"_GW{current_gw + horizon - 1}"
        
        # 1. CLASSIFIER: Probability of Playing
        clf_model = self.models.get(f'classifier_h{horizon}')
        prob_play = 1.0 # Default to 1.0 if no classifier
        
        if clf_model:
            X_clf = self._get_aligned_features(clf_model, X)
            if hasattr(clf_model, "predict_proba"):
                prob_play = clf_model.predict_proba(X_clf)[:, 1]
                self.df[f'prob_play{suffix}'] = prob_play
            
        # 2. REGRESSOR: Points (Conditional on playing)
        reg_model = self.models.get(f'regressor_h{horizon}')
        cond_points = 0.0
        
        if reg_model:
            X_reg = self._get_aligned_features(reg_model, X)
            if hasattr(reg_model, "predict"):
                cond_points = reg_model.predict(X_reg)
                # Clip negative predictions (impossible in FPL context usually)
                cond_points = np.maximum(cond_points, 0)
        
        # 3. COMBINE: Expected Points 
        self.df[f'xP{suffix}'] = prob_play * cond_points
    
    def generate_report(self, current_gw: int) -> pd.DataFrame:
        """
        Aggregates predictions into a transfer planner.
        """
        X = self.df[self.features].fillna(0)
        
        horizons = [1, 2, 3]
        for h in horizons:
            self._predict_horizon(h, X, current_gw)
            
        xp_cols = [f'xP_GW{current_gw + h - 1}' for h in horizons]
        available_xp_cols = [col for col in xp_cols if col in self.df.columns]
        
        if available_xp_cols:
            self.df['xP_3GW_Total'] = self.df[available_xp_cols].sum(axis=1)
        else:
            self.df['xP_3GW_Total'] = 0.0
        
        # Metadata
        base_meta = ['name', 'team', 'position', 'value', 'form_score', 'ict_index']
        optional_meta = ['minutes', 'opponent_team', 'selected', 'web_name', 'id']
        meta_cols = base_meta + [c for c in optional_meta if c in self.df.columns]
        
        # Prob Play columns are useful for the user to see risk
        prob_cols = [c for c in self.df.columns if 'prob_play' in c]
        
        output_cols = meta_cols + available_xp_cols + ['xP_3GW_Total'] + prob_cols
        
        
        # Only filtering dead assets (Value < 3.9m) to reduce noise
        if 'value' in self.df.columns:
            active_players = self.df[self.df['value'] >= 39].copy()
        else:
            active_players = self.df.copy()
            
        return active_players[output_cols].sort_values('xP_3GW_Total', ascending=False)


def train_jit_models(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Trains Two-Stage models (Classifier + Regressor) on the fly.
    """
    logger.info(f"INITIATING JIT TRAINING (Features: {len(features)})")
    models = {}
    
    # Sort for correct time-shifting
    df_train = df.sort_values(['element', 'season', 'GW']).copy()
    
    for h in [1, 2, 3]:
        # Targets
        target_pts = f'target_h{h}'
        target_mins = f'minutes_h{h}'
        
        # Shift future values back to current row
        df_train[target_pts] = df_train.groupby('element')['total_points'].shift(-h)
        df_train[target_mins] = df_train.groupby('element')['minutes'].shift(-h)
        
        # Drop rows where we don't know the future (end of season)
        valid_mask = df_train[target_pts].notna()
        data = df_train.loc[valid_mask].copy()
        
        X = data[features]
        
        # Target: Did they play > 0 minutes?
        y_class = (data[target_mins] > 0).astype(int)
        
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        clf.fit(X, y_class)
        models[f'classifier_h{h}'] = clf
        
        # Train ONLY on players who actually played
        play_mask = y_class == 1
        X_reg = X.loc[play_mask]
        y_reg = data.loc[play_mask, target_pts]
        
        reg = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        reg.fit(X_reg, y_reg)
        models[f'regressor_h{h}'] = reg
        
        logger.info(f"  [OK] Horizon {h}: Trained Classifier ({len(X)} samples) & Regressor ({len(X_reg)} samples)")
        
    return models

def get_model_features(df: pd.DataFrame) -> List[str]:
    """Identifies feature columns."""
    exclude_patterns = [
        'element', 'name', 'position', 'team', 'season', 'GW',
        'total_points', 'fixture', 'kickoff_time', 'opponent_team',
        'match_score', 'value', 'selected', 'transfers_', 'was_home',
        'team_h_score', 'team_a_score', 'target_', 'xP_', 'prob_', 'minutes'
    ]
    
    feature_cols = []
    for col in df.columns:
        if any(pattern in col for pattern in exclude_patterns):
            # Keep rolling/lagged minutes, exclude current minutes
            if 'minutes' in col and ('roll' in col or 'lag' in col or 'mean' in col):
                pass 
            else:
                continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
            
    return feature_cols