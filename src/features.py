"""
Feature Engineering Module
===========================

This module transforms preprocessed match-level data into predictive features
that capture temporal dynamics, opponent context, and position-specific patterns.

Mathematical Framework:
Feature vector X(i,t) for player i at gameweek t incorporates:
1. Rolling statistics with exponential temporal weighting
2. Lag features capturing immediate recent performance
3. Variance metrics quantifying performance volatility
4. Form trajectory indicators (improving vs declining)
5. Price-efficiency metrics identifying value opportunities

Temporal Guarantee:
All features X(i,t) depend exclusively on data from gameweeks <= t-1,
ensuring no information leakage during training or prediction.

Author: Splinter
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR = BASE_DIR / "data" / "features"

# Core performance metrics to engineer
PERFORMANCE_METRICS = [
    'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
    'ict_index', 'influence', 'creativity', 'threat', 'bps',
    'saves', 'goals_conceded', 'bonus'
]

# Expected goals metrics (available from 2022-23 onward)
EXPECTED_METRICS = [
    'expected_goals', 'expected_assists', 
    'expected_goal_involvements', 'expected_goals_conceded'
]

# Rolling windows for temporal aggregation
ROLLING_WINDOWS = [3, 5, 10]

# ============================================================================
# ROLLING STATISTICS
# ============================================================================

def calculate_rolling_mean(df: pd.DataFrame, 
                          metrics: List[str], 
                          windows: List[int]) -> pd.DataFrame:
    """
    Calculates rolling mean statistics for specified metrics.
    
    Mathematical Formulation:
    For metric m at time t with window size w:
        rolling_mean(m,t) = (1/w) * Σ(k=1 to w) m(t-k)
    
    The shift(1) ensures we use only past data - the rolling calculation
    at time t uses gameweeks [t-w, t-1], never including time t itself.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe sorted by player and gameweek
    metrics : List[str]
        Performance metrics to aggregate
    windows : List[int]
        Window sizes in gameweeks
        
    Returns
    -------
    pd.DataFrame
        Dataframe with rolling mean columns added
    """
    df_result = df.copy()
    
    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found, skipping...")
            continue
            
        for window in windows:
            col_name = f'{metric}_roll_mean_{window}gw'
            
            # Calculate rolling mean per player, then shift to prevent leakage
            df_result[col_name] = (
                df_result.groupby('element')[metric]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
                .shift(1)  # Critical: shift to use only past data
            )
    
    return df_result


def calculate_rolling_std(df: pd.DataFrame, 
                         metrics: List[str], 
                         windows: List[int]) -> pd.DataFrame:
    """
    Calculates rolling standard deviation to quantify performance volatility.
    
    Mathematical Rationale:
    Variance itself is informative - high variance forwards indicate boom-bust
    potential, low variance defenders suggest consistent floor. Standard deviation
    σ at time t quantifies recent performance dispersion:
        σ(m,t) = sqrt((1/w) * Σ(k=1 to w) (m(t-k) - μ(t))²)
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe sorted by player and gameweek
    metrics : List[str]
        Performance metrics to analyze
    windows : List[int]
        Window sizes in gameweeks
        
    Returns
    -------
    pd.DataFrame
        Dataframe with rolling std columns added
    """
    df_result = df.copy()
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        for window in windows:
            col_name = f'{metric}_roll_std_{window}gw'
            
            df_result[col_name] = (
                df_result.groupby('element')[metric]
                .rolling(window=window, min_periods=2)  # Need >= 2 for std
                .std()
                .reset_index(level=0, drop=True)
                .shift(1)
            )
    
    return df_result


def calculate_exponential_weighted_mean(df: pd.DataFrame,
                                       metrics: List[str],
                                       span: int = 5) -> pd.DataFrame:
    """
    Calculates exponentially weighted moving average (EWMA).
    
    Mathematical Formulation:
    EWMA assigns exponentially declining weights to historical observations:
        EWMA(t) = α * x(t) + (1-α) * EWMA(t-1)
    
    where α = 2/(span+1) is the smoothing factor.
    
    EWMA responds faster to recent changes than simple moving average while
    still incorporating longer historical context with declining influence.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe sorted by player and gameweek
    metrics : List[str]
        Performance metrics to smooth
    span : int
        Span parameter controlling decay rate (higher = slower decay)
        
    Returns
    -------
    pd.DataFrame
        Dataframe with EWMA columns added
    """
    df_result = df.copy()
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        col_name = f'{metric}_ewma_{span}gw'
        
        df_result[col_name] = (
            df_result.groupby('element')[metric]
            .ewm(span=span, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
            .shift(1)
        )
    
    return df_result


# ============================================================================
# LAG FEATURES
# ============================================================================

def calculate_lag_features(df: pd.DataFrame, 
                          metrics: List[str], 
                          lags: List[int] = [1, 2]) -> pd.DataFrame:
    """
    Creates lagged features capturing recent specific performances.
    
    Mathematical Justification:
    While rolling averages capture trends, lag features preserve information
    about specific recent gameweeks. A player who scored 15 points last week
    provides different signal than one who averaged 5 points over three weeks.
    
    Lag-1 (previous gameweek) often exhibits autocorrelation with current
    performance, particularly for form-dependent metrics like points and ICT.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe sorted by player and gameweek
    metrics : List[str]
        Performance metrics to lag
    lags : List[int]
        Lag periods in gameweeks
        
    Returns
    -------
    pd.DataFrame
        Dataframe with lag columns added
    """
    df_result = df.copy()
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        for lag in lags:
            col_name = f'{metric}_lag_{lag}gw'
            
            df_result[col_name] = (
                df_result.groupby('element')[metric]
                .shift(lag)
            )
    
    return df_result


# ============================================================================
# FORM TRAJECTORY FEATURES
# ============================================================================

def calculate_form_trajectory(df: pd.DataFrame, 
                             metrics: List[str]) -> pd.DataFrame:
    """
    Constructs features indicating whether form is improving or declining.
    
    Mathematical Definition:
    Form trajectory compares short-term vs medium-term averages:
        trajectory(t) = rolling_mean_3gw(t) - rolling_mean_5gw(t)
    
    Positive values indicate recent improvement (upward trajectory).
    Negative values indicate recent decline (downward trajectory).
    
    This captures momentum that pure rolling averages miss - a player averaging
    4 points over 5 weeks but 6 points over last 3 weeks shows positive momentum.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with rolling mean features already calculated
    metrics : List[str]
        Base metrics to analyze
        
    Returns
    -------
    pd.DataFrame
        Dataframe with trajectory columns added
    """
    df_result = df.copy()
    
    for metric in metrics:
        short_window = f'{metric}_roll_mean_3gw'
        long_window = f'{metric}_roll_mean_5gw'
        
        if short_window in df.columns and long_window in df.columns:
            trajectory_col = f'{metric}_trajectory'
            df_result[trajectory_col] = (
                df_result[short_window] - df_result[long_window]
            )
    
    return df_result


# ============================================================================
# PRICE EFFICIENCY FEATURES
# ============================================================================

def calculate_price_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs price-performance interaction features.
    
    Mathematical Framework:
    Price efficiency measures points delivered per unit cost:
        efficiency(i,t) = points_rolling_mean(i,t) / value(i,t)
    
    This identifies players outperforming their market valuation, which often
    represents either temporary underpricing or sustained value opportunities.
    
    High efficiency suggests either undervaluation (exploitable edge) or
    exceptional form likely to regress (mean reversion risk). The model must
    learn which interpretation applies based on other features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with rolling statistics and player values
        
    Returns
    -------
    pd.DataFrame
        Dataframe with price efficiency columns added
    """
    df_result = df.copy()
    
    if 'value' not in df.columns:
        logger.warning("Value column not found, skipping price efficiency features")
        return df_result
    
    # Points per million over different windows
    for window in [3, 5]:
        points_col = f'total_points_roll_mean_{window}gw'
        
        if points_col in df.columns:
            efficiency_col = f'points_per_million_{window}gw'
            
            # Avoid division by zero
            df_result[efficiency_col] = np.where(
                df_result['value'] > 0,
                df_result[points_col] / (df_result['value'] / 10.0),  # Convert to millions
                0
            )
    
    # Form-adjusted price (inverse efficiency)
    if 'total_points_roll_mean_5gw' in df.columns:
        df_result['value_per_point'] = np.where(
            df_result['total_points_roll_mean_5gw'] > 0,
            (df_result['value'] / 10.0) / df_result['total_points_roll_mean_5gw'],
            df_result['value'] / 10.0
        )
    
    return df_result


# ============================================================================
# POSITION-SPECIFIC FEATURES
# ============================================================================

def calculate_position_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs features that emphasize position-relevant metrics.
    
    Mathematical Rationale:
    Different positions exhibit distinct scoring mechanisms requiring
    specialized feature engineering:
    
    GK: Clean sheet probability, save volume, opponent attacking strength
    DEF: Clean sheet probability, attacking returns, bonus point frequency
    MID: All-around involvement (ICT), goal contributions, versatility
    FWD: Goal threat, conversion efficiency, penalty box presence
    
    We create position-weighted composite features that emphasize the metrics
    most predictive for each position category.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with position labels and base statistics
        
    Returns
    -------
    pd.DataFrame
        Dataframe with position-specific composite features
    """
    df_result = df.copy()
    
    if 'position' not in df.columns:
        logger.warning("Position column not found, skipping position-specific features")
        return df_result
    
    # Goalkeeper composite: Clean sheets + saves
    gk_mask = df_result['position'] == 'GK'
    if 'clean_sheets_roll_mean_5gw' in df.columns and 'saves_roll_mean_5gw' in df.columns:
        df_result['gk_composite'] = 0.0
        df_result.loc[gk_mask, 'gk_composite'] = (
            df_result.loc[gk_mask, 'clean_sheets_roll_mean_5gw'] * 4 +  # Clean sheet worth ~4 points
            df_result.loc[gk_mask, 'saves_roll_mean_5gw'] * 0.33  # Save worth ~1/3 point
        )
    
    # Defender composite: Clean sheets + attacking threat
    def_mask = df_result['position'] == 'DEF'
    if 'clean_sheets_roll_mean_5gw' in df.columns and 'threat_roll_mean_5gw' in df.columns:
        df_result['def_composite'] = 0.0
        df_result.loc[def_mask, 'def_composite'] = (
            df_result.loc[def_mask, 'clean_sheets_roll_mean_5gw'] * 4 +
            df_result.loc[def_mask, 'threat_roll_mean_5gw'] * 0.05  # Normalize threat scale
        )
    
    # Forward composite: Goal threat + conversion
    fwd_mask = df_result['position'] == 'FWD'
    if 'threat_roll_mean_5gw' in df.columns and 'goals_scored_roll_mean_5gw' in df.columns:
        df_result['fwd_composite'] = 0.0
        df_result.loc[fwd_mask, 'fwd_composite'] = (
            df_result.loc[fwd_mask, 'threat_roll_mean_5gw'] * 0.05 +
            df_result.loc[fwd_mask, 'goals_scored_roll_mean_5gw'] * 4  # Goal worth ~4-6 points
        )
    
    # Midfielder composite: Balanced ICT
    mid_mask = df_result['position'] == 'MID'
    if 'ict_index_roll_mean_5gw' in df.columns:
        df_result['mid_composite'] = 0.0
        df_result.loc[mid_mask, 'mid_composite'] = (
            df_result.loc[mid_mask, 'ict_index_roll_mean_5gw'] * 0.3
        )
    
    return df_result


# ============================================================================
# MATCH CONTEXT FEATURES
# ============================================================================

def calculate_match_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs features capturing match-level context.
    
    Features:
    - Home advantage indicator and interaction with form
    - Recent goal involvement ratio (goals + assists) / total_points
    - Bonus point frequency (bonus earned per match with minutes)
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with match-level data
        
    Returns
    -------
    pd.DataFrame
        Dataframe with context features added
    """
    df_result = df.copy()
    
    # Home form vs Away form
    if 'was_home' in df.columns and 'total_points_roll_mean_5gw' in df.columns:
        df_result['home_form_interaction'] = (
            df_result['was_home'].astype(int) * df_result['total_points_roll_mean_5gw']
        )
    
    # Goal involvement rate (what fraction of points come from goals/assists)
    if all(col in df.columns for col in ['goals_scored_roll_mean_5gw', 
                                          'assists_roll_mean_5gw', 
                                          'total_points_roll_mean_5gw']):
        df_result['goal_involvement_rate'] = np.where(
            df_result['total_points_roll_mean_5gw'] > 0,
            (df_result['goals_scored_roll_mean_5gw'] * 4 + 
             df_result['assists_roll_mean_5gw'] * 3) / df_result['total_points_roll_mean_5gw'],
            0
        )
    
    # Bonus frequency
    if 'bonus_roll_mean_5gw' in df.columns and 'minutes_roll_mean_5gw' in df.columns:
        df_result['bonus_frequency'] = np.where(
            df_result['minutes_roll_mean_5gw'] > 45,  # Regular starters
            df_result['bonus_roll_mean_5gw'],
            0
        )
    
    return df_result


# ============================================================================
# TARGET VARIABLE CONSTRUCTION
# ============================================================================

def create_target_variables(df: pd.DataFrame, 
                           horizons: List[int] = [3, 4]) -> pd.DataFrame:
    """
    Creates forward-looking cumulative point targets.
    
    Mathematical Definition:
    For player i at gameweek t with forecast horizon h:
        Y(i,t,h) = Σ(k=1 to h) total_points(i, t+k)
    
    CRITICAL TEMPORAL NOTE:
    These targets represent FUTURE information relative to gameweek t.
    They can only be used for:
    1. Validation on historical data where future is known
    2. Current prediction where we are forecasting unknown future
    
    They must NEVER appear as features during training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe sorted temporally
    horizons : List[int]
        Forecast horizons in gameweeks
        
    Returns
    -------
    pd.DataFrame
        Dataframe with target columns added
    """
    df_result = df.copy()
    
    for horizon in horizons:
        target_col = f'target_points_{horizon}gw'
        
        # Calculate forward-looking sum per player
        df_result[target_col] = (
            df_result.groupby('element')['total_points']
            .rolling(window=horizon, min_periods=1)
            .sum()
            .shift(-horizon + 1)  # Align with current gameweek
            .reset_index(level=0, drop=True)
        )
    
    return df_result


# ============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ============================================================================

def engineer_features(df: pd.DataFrame, 
                     include_targets: bool = True) -> pd.DataFrame:
    """
    Executes complete feature engineering pipeline.
    
    Pipeline Stages:
    1. Rolling statistics (mean, std, EWMA)
    2. Lag features
    3. Form trajectory indicators
    4. Price efficiency metrics
    5. Position-specific composites
    6. Match context features
    7. Target variables (if include_targets=True)
    
    Temporal Guarantee:
    All features use .shift(1) or equivalent to ensure they depend only on
    data from gameweeks prior to the current observation, preventing leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed match-level data sorted temporally
    include_targets : bool
        Whether to construct target variables
        
    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features
    """
    logger.info("="*80)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*80)
    
    # Validate temporal sorting
    df = df.sort_values(['season', 'GW', 'element']).reset_index(drop=True)
    logger.info(f"Input data: {len(df):,} rows × {len(df.columns)} columns")
    
    # Stage 1: Rolling statistics
    logger.info("\n1. Computing rolling statistics...")
    available_metrics = [m for m in PERFORMANCE_METRICS if m in df.columns]
    available_xg = [m for m in EXPECTED_METRICS if m in df.columns]
    all_metrics = available_metrics + available_xg
    
    df = calculate_rolling_mean(df, all_metrics, ROLLING_WINDOWS)
    df = calculate_rolling_std(df, all_metrics, ROLLING_WINDOWS)
    df = calculate_exponential_weighted_mean(df, all_metrics, span=5)
    
    rolling_features = [c for c in df.columns if any(x in c for x in ['_roll_', '_ewma_'])]
    logger.info(f"   Created {len(rolling_features)} rolling features")
    
    # Stage 2: Lag features
    logger.info("\n2. Computing lag features...")
    df = calculate_lag_features(df, all_metrics, lags=[1, 2])
    lag_features = [c for c in df.columns if '_lag_' in c]
    logger.info(f"   Created {len(lag_features)} lag features")
    
    # Stage 3: Form trajectory
    logger.info("\n3. Computing form trajectory indicators...")
    df = calculate_form_trajectory(df, all_metrics)
    trajectory_features = [c for c in df.columns if '_trajectory' in c]
    logger.info(f"   Created {len(trajectory_features)} trajectory features")
    
    # Stage 4: Price efficiency
    logger.info("\n4. Computing price efficiency metrics...")
    df = calculate_price_efficiency(df)
    price_features = [c for c in df.columns if 'per_million' in c or 'value_per' in c]
    logger.info(f"   Created {len(price_features)} price features")
    
    # Stage 5: Position-specific features
    logger.info("\n5. Computing position-specific composites...")
    df = calculate_position_specific_features(df)
    position_features = [c for c in df.columns if '_composite' in c]
    logger.info(f"   Created {len(position_features)} position features")
    
    # Stage 6: Match context features
    logger.info("\n6. Computing match context features...")
    df = calculate_match_context_features(df)
    context_features = [c for c in df.columns if any(x in c for x in 
                        ['_interaction', '_rate', '_frequency'])]
    logger.info(f"   Created {len(context_features)} context features")
    
    # Stage 7: Target variables
    if include_targets:
        logger.info("\n7. Creating target variables...")
        df = create_target_variables(df, horizons=[3, 4])
        logger.info("   Created target_points_3gw, target_points_4gw")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("="*80)
    logger.info(f"Final dimensions: {len(df):,} rows × {len(df.columns)} columns")
    
    # Feature count by category
    feature_summary = {
        'Rolling Statistics': len(rolling_features),
        'Lag Features': len(lag_features),
        'Trajectory Features': len(trajectory_features),
        'Price Efficiency': len(price_features),
        'Position-Specific': len(position_features),
        'Match Context': len(context_features)
    }
    
    logger.info("\nFeature Summary by Category:")
    for category, count in feature_summary.items():
        logger.info(f"  {category:25s}: {count:3d} features")
    
    return df


def load_and_engineer_features(save: bool = True) -> pd.DataFrame:
    """
    Loads preprocessed data and executes feature engineering pipeline.
    
    This is the main entry point for feature engineering in production workflows.
    
    Parameters
    ----------
    save : bool
        Whether to save engineered features to disk
        
    Returns
    -------
    pd.DataFrame
        Fully engineered dataset ready for modeling
    """
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    preprocessed_file = PROCESSED_DIR / "fpl_unified_preprocessed.csv"
    
    if not preprocessed_file.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {preprocessed_file}. "
            "Run src/preprocess.py first."
        )
    
    df = pd.read_csv(preprocessed_file)
    logger.info(f"Loaded {len(df):,} observations")
    
    # Engineer features
    df_features = engineer_features(df, include_targets=True)
    
    # Save if requested
    if save:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        output_file = FEATURES_DIR / "fpl_engineered_features.csv"
        df_features.to_csv(output_file, index=False)
        logger.info(f"\nEngineered features saved to: {output_file}")
    
    return df_features


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    """
    Execute feature engineering pipeline from command line.
    
    Usage:
        python src/features.py
    
    Output:
        data/features/fpl_engineered_features.csv
    """
    try:
        df = load_and_engineer_features(save=True)
        logger.info("\n✓ FEATURE ENGINEERING COMPLETE")
    except Exception as e:
        logger.error(f"\n✗ FEATURE ENGINEERING FAILED: {str(e)}")
        raise