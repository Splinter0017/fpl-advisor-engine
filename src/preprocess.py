"""
Data Preprocessing and Schema Harmonization Module
===================================================

This module transforms raw FPL data from multiple sources into a unified schema
suitable for feature engineering and modeling. It handles:
- Schema inconsistencies between seasons
- Column renaming and standardization
- Duplicate record resolution
- Position and team label harmonization
- Missing expected goals metric handling

Mathematical Foundation:
The preprocessing establishes a canonical feature space X ∈ R^(n×p) where n represents
player-match observations and p represents standardized features. This transformation
ensures temporal consistency required for valid time series modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

HISTORICAL_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
CURRENT_SEASON = "2025-26"

# Canonical position mapping (element_type to position)
POSITION_MAP = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

# Core columns that must exist in final schema
CORE_COLUMNS = [
    'element', 'name', 'position', 'team', 'GW', 'season',
    'total_points', 'minutes', 'goals_scored', 'assists', 
    'clean_sheets', 'bonus', 'bps',
    'influence', 'creativity', 'threat', 'ict_index',
    'opponent_team', 'was_home', 'kickoff_time',
    'saves', 'goals_conceded', 'yellow_cards', 'red_cards',
    'own_goals', 'penalties_missed', 'penalties_saved',
    'value', 'selected', 'transfers_in', 'transfers_out', 'transfers_balance'
]

# Expected goals columns (available only from 2022-23 onward)
EXPECTED_COLS = [
    'expected_goals', 'expected_assists', 
    'expected_goal_involvements', 'expected_goals_conceded'
]

# Defensive columns (available only in 2025-26)
DEFENSIVE_COLS = [
    'tackles', 'clearances_blocks_interceptions', 
    'recoveries', 'defensive_contribution'
]

# ============================================================================
# SCHEMA HARMONIZATION FUNCTIONS
# ============================================================================

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names across different data sources.
    
    The API uses 'round' while historical data uses 'GW'. We standardize to 'GW'
    as the canonical gameweek identifier. Similarly, 'GKP' is mapped to 'GK'.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with potentially inconsistent column names
        
    Returns
    -------
    pd.DataFrame
        Dataframe with standardized column names
    """
    df = df.copy()
    
    # Standardize gameweek column
    if 'round' in df.columns and 'GW' not in df.columns:
        df['GW'] = df['round']
        df = df.drop(columns=['round'])
        logger.info("Standardized 'round' -> 'GW'")
    
    return df


def standardize_position_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonizes position labels to canonical four-category system.
    
    Resolves the GK vs GKP inconsistency by mapping all goalkeeper variants
    to the canonical 'GK' label. This ensures consistent position-based
    feature engineering and modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with position column potentially containing inconsistent labels
        
    Returns
    -------
    pd.DataFrame
        Dataframe with standardized position labels
    """
    df = df.copy()
    
    if 'position' in df.columns:
        # Map all goalkeeper variants to canonical GK
        position_corrections = {'GKP': 'GK'}
        df['position'] = df['position'].replace(position_corrections)
        
        n_corrected = (df['position'] == 'GK').sum()
        logger.info(f"Standardized position labels: {n_corrected} goalkeepers")
    
    return df


def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolves duplicate player-gameweek records.
    
    Duplicate resolution strategy:
    1. If duplicates have identical data, keep first occurrence
    2. If duplicates differ in fixture ID, they represent legitimate separate matches
    3. If duplicates differ in statistics for same fixture, keep record with more minutes
    
    The mathematical justification: we require unique observations in feature space
    where each (player, gameweek) tuple maps to exactly one feature vector.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe potentially containing duplicate records
        
    Returns
    -------
    pd.DataFrame
        Dataframe with duplicates resolved
    """
    df = df.copy()
    
    initial_rows = len(df)
    
    # Identify duplicates based on player-gameweek combination
    duplicate_mask = df.duplicated(subset=['element', 'GW'], keep=False)
    n_duplicates = duplicate_mask.sum()
    
    if n_duplicates == 0:
        logger.info("No duplicate records found")
        return df
    
    logger.info(f"Found {n_duplicates} duplicate player-gameweek records")
    
    # Check if duplicates represent different fixtures
    if 'fixture' in df.columns:
        duplicates_df = df[duplicate_mask].copy()
        
        # Group by player-gameweek and check fixture variance
        fixture_variance = duplicates_df.groupby(['element', 'GW'])['fixture'].nunique()
        multiple_fixtures = (fixture_variance > 1).sum()
        
        if multiple_fixtures > 0:
            logger.info(f"{multiple_fixtures} player-gameweek pairs have multiple fixtures (legitimate)")
            # These are legitimate - players who played multiple matches in one gameweek
            # We keep all records for these cases
            
            # Only remove duplicates where fixture is same
            df = df.sort_values(['element', 'GW', 'fixture', 'minutes'], ascending=[True, True, True, False])
            df = df.drop_duplicates(subset=['element', 'GW', 'fixture'], keep='first')
        else:
            # All duplicates are for same fixture - keep record with most minutes
            df = df.sort_values(['element', 'GW', 'minutes'], ascending=[True, True, False])
            df = df.drop_duplicates(subset=['element', 'GW'], keep='first')
    else:
        # No fixture column - simple deduplication keeping first
        df = df.drop_duplicates(subset=['element', 'GW'], keep='first')
    
    final_rows = len(df)
    removed = initial_rows - final_rows
    logger.info(f"Removed {removed} duplicate records ({removed/initial_rows*100:.2f}%)")
    
    return df


def handle_expected_goals_metrics(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Handles expected goals metrics that are missing in early seasons.
    
    Strategy for 2021-22 season (pre-xG era):
    - Create columns filled with NaN to maintain schema consistency
    - During feature engineering, we will either:
      a) Impute using regression models trained on later seasons
      b) Create separate feature sets that work with/without xG
    
    We do NOT drop 2021-22 because it contains 25k observations with valuable
    signal in other features. The missing xG data is informative missingness
    that we can model explicitly.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe potentially missing expected goals columns
    season : str
        Season identifier to determine expected data availability
        
    Returns
    -------
    pd.DataFrame
        Dataframe with expected goals columns (filled with NaN if unavailable)
    """
    df = df.copy()
    
    # For seasons before 2022-23, expected goals metrics don't exist
    if season == "2021-22":
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = np.nan
        logger.info(f"Season {season}: Added expected goals columns as NaN (pre-xG era)")
    
    return df


def handle_defensive_metrics(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Handles defensive metrics available only in current season.
    
    These columns provide granular defensive statistics but didn't exist in
    historical data. For historical seasons, we create columns filled with NaN.
    Our models will learn patterns from current season data only, or we may
    choose to exclude these features from the initial model version.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe potentially missing defensive metric columns
    season : str
        Season identifier
        
    Returns
    -------
    pd.DataFrame
        Dataframe with defensive columns (filled with NaN if unavailable)
    """
    df = df.copy()
    
    # Only 2025-26 has these metrics
    if season != CURRENT_SEASON:
        for col in DEFENSIVE_COLS:
            if col not in df.columns:
                df[col] = np.nan
        logger.info(f"Season {season}: Added defensive columns as NaN (not available)")
    
    return df


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived fields that will be useful for feature engineering.
    
    CRITICAL: Removed points_per_minute to avoid data leakage.
    
    Derived fields include:
    - match_score: String representation of final score
    - is_starter: Boolean indicating if player started (>= 60 minutes heuristic)
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with additional derived fields
    """
    df = df.copy()
    
    # Match score string for reference
    if 'team_h_score' in df.columns and 'team_a_score' in df.columns:
        df['match_score'] = df['team_h_score'].astype(str) + '-' + df['team_a_score'].astype(str)
    
    # Starter indicator (heuristic: 60+ minutes)
    if 'minutes' in df.columns:
        df['is_starter'] = (df['minutes'] >= 60).astype(int)
    
    return df

def harmonize_team_identifiers(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Maps integer Team IDs to string Team Names using the season's reference file.
    
    This is critical for 'opponent_team', which often comes as an integer ID
    (e.g., 14) that must be linked to team-level metadata (e.g., 'Man City').
    
    CRITICAL FIX: data.py now handles 'team' column mapping, so we only need
    to handle 'opponent_team' here.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'opponent_team' column (integers)
    season : str
        Season identifier to locate the correct teams.csv lookup
        
    Returns
    -------
    pd.DataFrame
        Dataframe with 'opponent_team_name' column
    """
    df = df.copy()
    
    # 1. Load the Lookup Table
    teams_file = RAW_DIR / f"{season}_teams.csv"
    
    if not teams_file.exists():
        logger.warning(f"Map file missing: {teams_file}. Opponent names cannot be resolved.")
        return df
        
    try:
        teams_df = pd.read_csv(teams_file)
        
        # Create Hash Map: ID -> Name
        name_col = 'name' if 'name' in teams_df.columns else 'team'
        id_map = pd.Series(teams_df[name_col].values, index=teams_df['id']).to_dict()
        
        # 2. Map Opponent Team IDs to Names
        if 'opponent_team' in df.columns:
            # Ensure it's numeric before mapping
            df['opponent_team'] = pd.to_numeric(df['opponent_team'], errors='coerce')
            df['opponent_team_name'] = df['opponent_team'].map(id_map)
            
            # Validation
            missing_opps = df['opponent_team_name'].isna().sum()
            if missing_opps > 0:
                logger.warning(f"Season {season}: {missing_opps} opponent IDs could not be mapped to names.")
        
        # 3. Validate Own Team Column (Should already be names from data.py)
        if 'team' in df.columns:
            if pd.api.types.is_numeric_dtype(df['team']):
                logger.warning(f"Season {season}: 'team' column is numeric - this should have been fixed in data.py")
                df['team'] = df['team'].map(id_map)
            else:
                logger.info(f"Season {season}: 'team' column verified as string names")

    except Exception as e:
        logger.error(f"Team mapping failed for {season}: {e}")
        
    return df

# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_season(season: str) -> Optional[pd.DataFrame]:
    """
    Preprocesses a single season's data through complete harmonization pipeline.
    
    Pipeline stages:
    1. Load raw data
    2. Standardize column names
    3. Standardize position labels
    4. Resolve duplicate records
    5. Handle expected goals metrics
    6. Handle defensive metrics
    7. Add derived fields (NO DATA LEAKAGE)
    8. Harmonize team identifiers
    9. Validate schema completeness
    
    Parameters
    ----------
    season : str
        Season identifier (e.g., "2021-22")
        
    Returns
    -------
    Optional[pd.DataFrame]
        Preprocessed dataframe, or None if file not found
    """
    logger.info(f"{'='*80}")
    logger.info(f"Processing Season: {season}")
    logger.info(f"{'='*80}")
    
    # Load raw data
    raw_file = RAW_DIR / f"{season}_merged_gw.csv"
    
    if not raw_file.exists():
        logger.warning(f"File not found: {raw_file}")
        return None
    
    df = pd.read_csv(raw_file)
    logger.info(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")
    
    # Apply preprocessing pipeline
    df = standardize_column_names(df)
    df = standardize_position_labels(df)
    df = resolve_duplicates(df)
    df = handle_expected_goals_metrics(df, season)
    df = handle_defensive_metrics(df, season)
    df = add_derived_fields(df)  # Now safe - no data leakage
    df = harmonize_team_identifiers(df, season)
    
    # Ensure season column exists
    df['season'] = season
    
    # Validate core columns exist
    missing_core = [col for col in CORE_COLUMNS if col not in df.columns]
    if missing_core:
        logger.warning(f"Missing core columns: {missing_core}")
    
    logger.info(f"Preprocessing complete: {len(df):,} rows retained")
    
    return df


def preprocess_all_seasons(save: bool = True) -> pd.DataFrame:
    """
    Preprocesses all seasons and merges into unified dataset.
    
    This function orchestrates the complete preprocessing pipeline across all
    available seasons, producing a single unified dataframe with consistent
    schema suitable for feature engineering and modeling.
    
    Parameters
    ----------
    save : bool, default=True
        If True, saves preprocessed data to disk
        
    Returns
    -------
    pd.DataFrame
        Unified preprocessed dataset spanning all seasons
    """
    logger.info("="*80)
    logger.info("PREPROCESSING ALL SEASONS")
    logger.info("="*80)
    
    all_seasons = []
    
    for season in HISTORICAL_SEASONS + [CURRENT_SEASON]:
        df_season = preprocess_season(season)
        if df_season is not None:
            all_seasons.append(df_season)
    
    if not all_seasons:
        raise ValueError("No data was successfully preprocessed")
    
    # Merge all seasons
    logger.info("\n" + "="*80)
    logger.info("MERGING SEASONS")
    logger.info("="*80)
    
    df_unified = pd.concat(all_seasons, ignore_index=True)
    
    logger.info(f"Unified dataset: {len(df_unified):,} rows × {len(df_unified.columns)} columns")
    logger.info(f"Temporal coverage: {df_unified['GW'].min()} to {df_unified['GW'].max()} gameweeks")
    logger.info(f"Seasons included: {sorted(df_unified['season'].unique())}")
    
    # Sort by temporal order
    df_unified = df_unified.sort_values(['season', 'GW', 'element']).reset_index(drop=True)
    
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_file = PROCESSED_DIR / "fpl_unified_preprocessed.csv"
        df_unified.to_csv(output_file, index=False)
        logger.info(f"\nSaved to: {output_file}")
    
    # Generate summary statistics
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*80)
    
    print("\nSeason Distribution:")
    print(df_unified['season'].value_counts().sort_index())
    
    print("\nPosition Distribution:")
    if 'position' in df_unified.columns:
        print(df_unified['position'].value_counts())
    
    print("\nTeam Column Validation:")
    if 'team' in df_unified.columns:
        print(f"  Data type: {df_unified['team'].dtype}")
        print(f"  Unique teams: {df_unified['team'].nunique()}")
        print(f"  Sample values: {df_unified['team'].dropna().head(5).tolist()}")
    
    print("\nMissing Values in Key Columns:")
    key_cols = ['total_points', 'minutes', 'position', 'team', 'expected_goals', 'expected_assists']
    available_key_cols = [col for col in key_cols if col in df_unified.columns]
    missing_summary = df_unified[available_key_cols].isna().sum()
    missing_pct = (missing_summary / len(df_unified) * 100).round(2)
    missing_df = pd.DataFrame({'Missing': missing_summary, 'Percent': missing_pct})
    print(missing_df)
    
    return df_unified


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    try:
        df = preprocess_all_seasons(save=True)
        logger.info("\nPREPROCESSING PIPELINE COMPLETE")
    except Exception as e:
        logger.error(f"\nPREPROCESSING FAILED: {str(e)}")
        raise