# Frontend Scaffold

This directory contains the initial Streamlit frontend scaffold for the Industrial Surface Defect Inspection Platform.

## Run Locally

Install the project requirements first:

```bash
python -m pip install -r requirements.txt
```

Then start the scaffold:

```bash
streamlit run frontend/streamlit_app.py
```

## What This Scaffold Does

- Shows the project title and current high-level status
- Provides sidebar navigation for the planned dashboard pages
- Presents a safe limitations banner
- Establishes the structure for reading existing JSON data contracts in the next phase

## What Is Not Implemented Yet

- Live upload/predict behavior
- API-backed inference
- Docker or docker-compose
- Full chart wiring from JSON data contracts
- Production or deployment claims

## Next Phase

The next phase is to wire the dashboard to the existing JSON data contracts in:

- `artifacts/frontend/track_a/`
- `artifacts/frontend/track_b/`
- `artifacts/frontend/detection/yolo_train_v0_2_0/`
