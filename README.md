# ATP Match Predictor

A Python side project that estimates tennis match win probabilities
using ATP rankings, overall Elo and surface-specific Elo.

Includes:
- Data preparation and historical Elo calculation
- Logistic regression and parameter tuning
- Temperature calibration
- A local Streamlit interface

## Setup

Create and activate a Python virtual environment, then run:

    python -m pip install -r requirements.txt

## Data and model files

CSV datasets, tuning results and trained model files are not included.
Data source: https://github.com/Tennismylife/TML-Database

The app requires locally generated:
- tennis_model.joblib
- players_snapshot.csv

The existing scripts prepare data, build Elo features, tune parameters
and train the model. Use matching settings when generating the model
and player snapshot.

## Run the app

    python -m streamlit run app.py

## Limitations

This is an experimental hobby project (meh)
The initial player snapshot covers January 2026, not live player form.
Elo updates use a 21-day delay because exact match dates are unavailable.
Predictions assume completed matches and exclude retirement outcomes.