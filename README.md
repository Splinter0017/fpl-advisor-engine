FPL-Advisor-Engine: Algorithmic Sports Trading System

Project Overview

An automated decision support system for Fantasy Premier League (FPL) that treats the game as a constrained resource optimization problem. This project moves beyond simple heuristics by implementing a "Cyber-Physical System" architecture: it digitizes real-world match events, processes them through predictive models, and delivers actionable trading strategies via an LLM Agent interface.

The system is built on a "Three-Brain" architecture designed to solve the specific challenges of sports analytics: Signal Extraction (ETL), Outcome Prediction (ML), and Resource Allocation (OR).

Architecture

graph TD
    subgraph "The Engineer (Data Pipeline)"
        A[FPL API / Vaastav Repo] -->|Raw JSON/CSV| B(ETL Pipeline / src/etl.py)
        B -->|Feature Engineering| C[Feature Store / data/processed]
    end
    
    subgraph "The Analyst (Prediction)"
        C -->|Historical Data| D{XGBoost Regressor}
        D -->|Future Fixtures| E(Predicted Points / src/predict.py)
    end
    
    subgraph "The Strategist (Optimization)"
        E -->|Predictions + Budget| F[MILP Solver / src/optimize.py]
        F -->|Knapsack Problem| G[Optimal Squad]
    end
    
    subgraph "The Interface (Deployment)"
        G -->|Sync| H[Google Sheets Dashboard]
        H -->|Context| I[Gemini AI Agent]
    end


The "Three-Brain" System

1. The Analyst (Predictive Modeling)

Algorithm: XGBoost Regressor (Gradient Boosting).

Training Data: 4+ Seasons of historical player-match data (>120,000 rows).

Feature Engineering:

Tactical Vulnerability (opp_def_strength_vs_pos): A dynamic matrix quantifying how many points specific opponents concede to specific positions (e.g., "Sheffield Utd weakness vs Left Wingers").

Luck Detection (xp_delta): Measures regression to the mean by comparing actual points against underlying stats (ICT Index proxy for xG/xA).

Team Strength Diff: Dynamic ELO-like rating difference between player's team and opponent.

Performance: Achieved a Mean Absolute Error (MAE) of ~1.04 points on the 2024-25 test set.

2. The Strategist (Operations Research)

Library: PuLP (Python Linear Programming).

Problem Formulation: Mixed-Integer Linear Programming (MILP) variant of the Knapsack Problem.

Constraints:

Budget $\le$ £100.0m.

Squad Size = 15 (11 Starters + 4 Bench).

Formation Rules: 2 GK, 5 DEF, 5 MID, 3 FWD.

Team Constraint: Max 3 players per Premier League club.

Objective: Maximize $\sum$ Predicted Points of the Starting XI while minimizing bench cost.

3. The Interface (Cloud Sync)

Automation: Python scripts automatically push the "Optimal Squad" and "Predictions" to a Google Sheet via Service Account authentication.

LLM Integration: A custom Gemini Gem is connected to this live sheet, acting as a "Forensic Advisor." It interprets the raw math into natural language strategy (e.g., "Salah is the captaincy pick because he faces a high-vulnerability defense, despite his high price.").

Quick Start

Prerequisites: Python 3.10+, Google Cloud Service Account JSON.

Clone the Repo

git clone [https://github.com/Splinter0017/fpl-advisor-engine.git](https://github.com/Splinter0017/fpl-advisor-engine.git)
cd fpl-advisor-engine


Environment Setup

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate


Install Dependencies

pip install -r requirements.txt


Configuration

Place your Google Cloud service_account.json key in the config/ folder.

(Optional) Adjust config/settings.yaml for custom budget constraints.

Run the Engine

# Step 1: Update Predictions (The Analyst)
python src/predict.py

# Step 2: Solve for Best Team (The Strategist)
python src/optimize.py
