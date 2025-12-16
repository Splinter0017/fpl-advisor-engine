"""
Advanced Feature Engineering Module
====================================

Features are constructed to capture temporal dependencies, contextual factors,
and position-specific patterns in player performance. The feature space is
designed to maximize predictive power while maintaining temporal validity.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# TEMPORAL FEATURES

def create_rolling_features(df: pd.DataFrame, 
                           windows: List[int] = [3, 5, 10],
                           stats: List[str] = ['total_points', 'minutes', 
                                               'goals_scored', 'assists',
                                               'expected_goals', 'expected_assists',
                                               'ict_index', 'bonus']) -> pd.DataFrame:
    """
    Creates rolling window statistics for key performance metrics.
    
    Mathematical Intuition:
    For a statistic s_t at gameweek t:
    
    Rolling Mean = (1/k) * sum(s_{t-k} to s_{t-1})
    Rolling Std  = sqrt((1/k) * sum((s_i - mean)^2))
    
    Short windows (3 GW): Capture recent form/momentum
    Medium windows (5 GW): Balance recency with stability
    Long windows (10 GW): Capture seasonal trends
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data with temporal ordering
    windows : List[int]
        Window sizes for rolling calculations
    stats : List[str]
        Statistics to aggregate
        
    Returns
    -------
    pd.DataFrame
        Original data augmented with rolling features
    """
    df = df.copy()
    
    # Sort by player and time
    df = df.sort_values(['element', 'season', 'GW'])
    
    # Group by player for temporal continuity
    grouped = df.groupby('element')
    
    for window in windows:
        for stat in stats:
            if stat not in df.columns:
                continue
                
            # Rolling mean
            df[f'{stat}_roll_mean_{window}'] = (
                grouped[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            
            # Rolling standard deviation (volatility)
            df[f'{stat}_roll_std_{window}'] = (
                grouped[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            )
            
            # Rolling maximum (peak performance)
            df[f'{stat}_roll_max_{window}'] = (
                grouped[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
            )
    
    logger.info(f"Created rolling features for windows: {windows}")
    return df


def create_ewm_features(df: pd.DataFrame,
                       spans: List[int] = [3, 5, 10],
                       stats: List[str] = ['total_points', 'minutes',
                                           'expected_goals', 'ict_index']) -> pd.DataFrame:
    """
    Creates exponentially weighted moving averages.
    
    Mathematical Intuition:
    EWMA_t = alpha * s_t + (1-alpha) * EWMA_{t-1}
    where alpha = 2/(span + 1)
    
    EWMA gives exponentially decaying weights to past observations,
    making recent performance more influential than distant past.
    
    Use Case:
    Captures "hot streaks" and "cold streaks" better than simple rolling mean.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data
    spans : List[int]
        Span parameters (higher = slower decay)
    stats : List[str]
        Statistics to weight
        
    Returns
    -------
    pd.DataFrame
        Data with EWMA features
    """
    df = df.copy()
    df = df.sort_values(['element', 'season', 'GW'])
    grouped = df.groupby('element')
    
    for span in spans:
        for stat in stats:
            if stat not in df.columns:
                continue
            
            df[f'{stat}_ewm_{span}'] = (
                grouped[stat]
                .transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())
            )
    
    logger.info(f"Created EWMA features for spans: {spans}")
    return df


def create_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates form indicators comparing recent performance to baseline.
    
    Mathematical Intuition:
    Form Score = (Recent Performance - Expected Performance) / Volatility
    
    This is a z-score of recent form.
    
    Interpretation:
    - Form > 1:  Player is overperforming (hot)
    - Form ~ 0:  Player is at expected level
    - Form < -1: Player is underperforming (cold)
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with rolling features already computed
        
    Returns
    -------
    pd.DataFrame
        Data with form indicators
    """
    df = df.copy()
    df = df.sort_values(['element', 'season', 'GW'])
    
    
    # Expanding mean and std (using only past data)
    df = df.sort_values(['element', 'season', 'GW'])
    df['season_mean_points'] = (
        df.groupby(['element', 'season'])['total_points']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    df['season_std_points'] = (
        df.groupby(['element', 'season'])['total_points']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).std())
    )
    
    # Form score (z-score of recent form)
    if 'total_points_ewm_5' in df.columns:
        df['form_score'] = (
            (df['total_points_ewm_5'] - df['season_mean_points']) / 
            (df['season_std_points'] + 1e-6)
        )
    
    # Minutes form (are they getting game time?)
    if 'minutes_roll_mean_3' in df.columns:
        df['minutes_form'] = df['minutes_roll_mean_3'] / 90.0
    
    logger.info("Created form features")
    return df


def create_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates streak counters for consecutive outcomes.
    
    Mathematical Intuition:
    Streak_t = Streak_{t-1} + 1 if condition holds, else 0
    
    Psychological Foundation:
    - Scoring streaks -> confidence -> better decision-making
    - Blank streaks -> pressure -> worse performance
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data
        
    Returns
    -------
    pd.DataFrame
        Data with streak features
    """
    df = df.copy()
    df = df.sort_values(['element', 'season', 'GW'])
    
    def compute_streak(series: pd.Series, threshold: float = 0) -> pd.Series:
        """Counts consecutive occurrences above threshold."""
        above_threshold = (series > threshold).astype(int)
        streak = above_threshold.groupby(
            (above_threshold != above_threshold.shift()).cumsum()
        ).cumsum()
        return streak.shift(1).fillna(0)
    
    grouped = df.groupby('element')
    
    # Scoring streak (consecutive GWs with points)
    df['scoring_streak'] = (
        grouped['total_points']
        .transform(lambda x: compute_streak(x, threshold=0))
    )
    
    # Blank streak (consecutive GWs with 0 points)
    df['blank_streak'] = (
        grouped['total_points']
        .transform(lambda x: compute_streak(-x, threshold=-0.1))
    )
    
    # Minutes streak (consecutive starts)
    if 'minutes' in df.columns:
        df['starter_streak'] = (
            grouped['minutes']
            .transform(lambda x: compute_streak(x, threshold=60))
        )
    
    logger.info("Created streak features")
    return df


# CONTEXTUAL FEATURES


def create_fixture_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates fixture difficulty features based on opponent strength.
    
    Mathematical Intuition:
    FDR = f(Opponent's Offensive Strength, Defensive Strength, Home/Away)
    
    A strong attacking team is harder for defenders.
    A strong defensive team is harder for attackers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with opponent_team_name column
        
    Returns
    -------
    pd.DataFrame
        Data with FDR features
    """
    df = df.copy()
    
    if 'opponent_team_name' not in df.columns:
        logger.warning("opponent_team_name column not found, skipping FDR features")
        return df
    
    # Compute team-level statistics (rolling across season)
    team_stats = df.groupby(['season', 'team', 'GW']).agg({
        'goals_scored': 'sum',
        'goals_conceded': 'sum',
        'clean_sheets': 'sum'
    }).reset_index()
    
    # Rolling team strength (last 5 gameweeks)
    team_stats = team_stats.sort_values(['team', 'season', 'GW'])
    team_grouped = team_stats.groupby(['team', 'season'])
    
    team_stats['team_goals_scored_last5'] = (
        team_grouped['goals_scored']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )
    
    team_stats['team_goals_conceded_last5'] = (
        team_grouped['goals_conceded']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )
    
    team_stats['team_clean_sheets_last5'] = (
        team_grouped['clean_sheets']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )
    
    # Merge opponent strength into main dataframe
    df = df.merge(
        team_stats[['season', 'team', 'GW', 
                    'team_goals_scored_last5', 
                    'team_goals_conceded_last5',
                    'team_clean_sheets_last5']],
        left_on=['season', 'opponent_team_name', 'GW'],
        right_on=['season', 'team', 'GW'],
        how='left',
        suffixes=('', '_opp')
    )
    
    # Drop duplicate team column
    if 'team_opp' in df.columns:
        df = df.drop(columns=['team_opp'])
    
    # Rename to clarify these are opponent stats
    df = df.rename(columns={
        'team_goals_scored_last5': 'opp_goals_scored_last5',
        'team_goals_conceded_last5': 'opp_goals_conceded_last5',
        'team_clean_sheets_last5': 'opp_clean_sheets_last5'
    })
    
    # FDR Score: Higher = Harder fixture
    # For attackers: Strong defense = hard
    # For defenders: Strong attack = hard
    df['fdr_defensive'] = df['opp_goals_conceded_last5'] / 5.0
    df['fdr_offensive'] = df['opp_goals_scored_last5'] / 5.0
    
    logger.info("Created fixture difficulty features")
    return df


def create_home_away_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features capturing home/away advantage.
    
    Statistical Foundation:
    Home advantage is well-documented in football:
    - Crowd support
    - Familiarity with pitch
    - Reduced travel fatigue
    
    Average home advantage: +33% in points
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'was_home' column
        
    Returns
    -------
    pd.DataFrame
        Data with home/away features
    """
    df = df.copy()
    
    if 'was_home' not in df.columns:
        logger.warning("was_home column not found, skipping home/away features")
        return df
    
    # Historical home/away split
    df = df.sort_values(['element', 'season', 'GW'])
    grouped = df.groupby('element')
    
    # Home performance baseline (expanding mean to avoid leakage)
    def safe_expanding_mean(group):
        home_mask = group['was_home'] == True
        home_points = group['total_points'].where(home_mask)
        return home_points.shift(1).expanding().mean()
    
    df['home_avg_points'] = (
        grouped.apply(safe_expanding_mean)
        .reset_index(level=0, drop=True)
    )
    
    # Away performance baseline
    def safe_expanding_mean_away(group):
        away_mask = group['was_home'] == False
        away_points = group['total_points'].where(away_mask)
        return away_points.shift(1).expanding().mean()
    
    df['away_avg_points'] = (
        grouped.apply(safe_expanding_mean_away)
        .reset_index(level=0, drop=True)
    )
    
    # Home/away delta
    df['home_away_delta'] = df['home_avg_points'] - df['away_avg_points']
    
    logger.info("Created home/away features")
    return df




# POSITION-SPECIFIC FEATURES

def create_position_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features tailored to each position's scoring pattern.
    
    Mathematical Intuition:
    Different positions have different covariance structures:
    
    Cov(CS, Points | GK) >> Cov(CS, Points | FWD)
    Cov(Goals, Points | FWD) >> Cov(Goals, Points | DEF)
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'position' column
        
    Returns
    -------
    pd.DataFrame
        Data with position-specific features
    """
    df = df.copy()
    
    if 'position' not in df.columns:
        logger.warning("position column not found, skipping position-specific features")
        return df
    
    # GK/DEF: Clean sheet dependency
    df['defensive_score'] = (
        df['clean_sheets'] * 4 +
        df.get('saves', 0) * 0.5 -
        df.get('goals_conceded', 0) * 0.5
    )
    
    # MID/FWD: Attacking involvement
    df['attacking_score'] = (
        df['goals_scored'] * 2 +
        df.get('assists', 0) * 1.5 +
        df.get('expected_goals', 0) * 1 +
        df.get('expected_assists', 0) * 1
    )
    
    # Position-weighted composite
    position_weights = {
        'GK': {'defensive': 1.0, 'attacking': 0.0},
        'DEF': {'defensive': 0.7, 'attacking': 0.3},
        'MID': {'defensive': 0.2, 'attacking': 0.8},
        'FWD': {'defensive': 0.0, 'attacking': 1.0}
    }
    
    df['position_adjusted_score'] = df.apply(
        lambda row: (
            row['defensive_score'] * position_weights.get(row['position'], {}).get('defensive', 0.5) +
            row['attacking_score'] * position_weights.get(row['position'], {}).get('attacking', 0.5)
        ),
        axis=1
    )
    
    logger.info("Created position-specific features")
    return df


# FEATURE ENGINEERING PIPELINE
class FeatureEngineer:
    """
    Orchestrates feature engineering transformations.
    
    Design Pattern: Pipeline
    - Each transformation is independent
    - Order matters (some features depend on others)
    - All transformations are temporal-safe (no leakage)
    """
    
    def __init__(self):
        self.transformations = [
            ('rolling', create_rolling_features),
            ('ewm', create_ewm_features),
            ('form', create_form_features),
            ('streaks', create_streak_features),
            ('fixture_difficulty', create_fixture_difficulty),
            ('home_away', create_home_away_features),
            ('position_specific', create_position_specific_features)
        ]
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all feature engineering transformations.
        
        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed data from preprocess.py
            
        Returns
        -------
        pd.DataFrame
            Feature-engineered dataset ready for modeling
        """
        logger.info("="*80)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("="*80)
        
        df_transformed = df.copy()
        initial_cols = len(df_transformed.columns)
        
        for name, transform_func in self.transformations:
            logger.info(f"\nApplying: {name}")
            try:
                cols_before = len(df_transformed.columns)
                df_transformed = transform_func(df_transformed)
                cols_after = len(df_transformed.columns)
                new_cols = cols_after - cols_before
                logger.info(f"  Added {new_cols} features (Total: {cols_after})")
            except Exception as e:
                logger.error(f"  Failed: {str(e)}")
                raise
        
        logger.info("\n" + "="*80)
        logger.info(f"FINAL FEATURE COUNT: {len(df_transformed.columns)}")
        logger.info(f"NEW FEATURES ADDED: {len(df_transformed.columns) - initial_cols}")
        logger.info("="*80)
        
        return df_transformed
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Returns list of engineered feature names."""
        engineered_patterns = [
            '_roll_', '_ewm_', 'form_', 'streak', 'fdr_', 
            'home_', 'away_', 'rest_', 'score', 'opp_'
        ]
        
        feature_cols = []
        for col in df.columns:
            if any(pattern in col for pattern in engineered_patterns):
                feature_cols.append(col)
        
        return feature_cols




# MAIN EXECUTION
if __name__ == "__main__":
    # Configuration
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    input_file = PROCESSED_DIR / "fpl_unified_preprocessed.csv"
    
    if not input_file.exists():
        logger.error(f"File not found: {input_file}")
        logger.error("Please run preprocess.py first")
        raise FileNotFoundError(f"Missing file: {input_file}")
    
    df = pd.read_csv(input_file)
    logger.info(f"Loaded: {len(df):,} rows x {len(df.columns)} columns")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Transform
    df_features = engineer.fit_transform(df)
    
    # Get feature names
    feature_cols = engineer.get_feature_names(df_features)
    logger.info(f"\nEngineered {len(feature_cols)} new features")
    
    # Save
    output_file = PROCESSED_DIR / "fpl_features_engineered.csv"
    df_features.to_csv(output_file, index=False)
    logger.info(f"\nSaved feature-engineered data to: {output_file}")
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("="*80)
    
    print(f"\nTotal columns: {len(df_features.columns)}")
    print(f"Engineered features: {len(feature_cols)}")
    print("\nMissing value summary (engineered features):")
    
    if feature_cols:
        missing_summary = df_features[feature_cols].isna().sum().sort_values(ascending=False).head(10)
        missing_pct = (missing_summary / len(df_features) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Count': missing_summary,
            'Percent': missing_pct
        })
        print(missing_df)
    
    logger.info("\nFEATURE ENGINEERING COMPLETE")