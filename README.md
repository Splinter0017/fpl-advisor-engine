# FPL Advisor Engine

## 1. Overview
The FPL Advisor Engine is a data science pipeline designed to optimize Fantasy Premier League decision-making through statistical modeling. It automates the entire lifecycle of FPL analysis: from data ingestion (historical & live) to advanced feature engineering and machine learning inference.

The system utilizes a **Two-Stage Model** to predict player performance over a 3-Gameweek horizon, solving the zero-inflation problem inherent in football data (i.e., players who don't play score zero points).

## 2. Mathematical Framework
The core inference engine utilizes a compound expectation derived from two distinct stochastic processes:

$$xP_{total} = P(Play) \times E[Points | Play]$$

1.  **Classifier:** A LightGBM Classifier estimates $P(Play)$, the probability that a player plays $>0$ minutes.
2.  **Regressor:** A LightGBM Regressor estimates $E[Points | Play]$, the expected points conditional on the player passing the hurdle.

This separation prevents the model from conflating "bad form" with "being benched," resulting in more accurate risk assessment for rotation-prone assets.

## 3. Project Architecture

The codebase adheres to a modular production-ready structure:

```text
fpl-advisor-engine/
├── data/                   # Data storage (ignored by Git)
│   ├── raw/                # Immutable raw CSVs from APIs
│   └── processed/          # Cleaned and feature-engineered datasets
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── outputs/                # Generated prediction reports (CSVs)
├── scripts/                # Executable entry points
│   └── prediction.py       # Main orchestration script for generating GW reports
├── src/                    # Source code library
│   ├── data.py             # Archive + Live API
│   ├── feature_engineering.py  # Temporal and contextual feature construction
│   ├── inference.py        # ML engine (Model implementation)
│   └── preprocess.py       # Data cleaning and unification
└── requirements.txt        # Dependency definitions