# Frontend Scaffold

This directory contains the initial Streamlit frontend scaffold for the Industrial Surface Defect Inspection Platform.

## Run Locally

Install the project requirements first:

```bash
python -m pip install -r requirements.txt
```

Start the API in one terminal:

```bash
uvicorn api.app.main:app --reload --port 8000
```

Then start the Streamlit app in another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

## What This Scaffold Does

- Shows the project title and current high-level status
- Loads and validates the existing JSON data contracts for Track A, Track B, and Detection
- Provides sidebar navigation for the planned dashboard pages
- Presents a safe limitations banner
- Includes a Track A upload / predict page that calls the local FastAPI endpoint
- Establishes the structure for reading existing JSON data contracts in the next phase

## What Is Not Implemented Yet

- YOLO and anomaly upload/predict
- Docker or docker-compose
- Full chart wiring from JSON data contracts
- Production or deployment claims

## Data Loading Phase

The current frontend phase validates that the following JSON data-contract folders are present and readable:

- `artifacts/frontend/track_a/`
- `artifacts/frontend/track_b/`
- `artifacts/frontend/detection/yolo_train_v0_2_0/`

The next frontend phase is to turn these validated payloads into dashboard sections and charts.

The upload / predict page expects the local API to be available at `http://localhost:8000` by default.

## Next Phase

The next frontend phase is to refine the local Track A upload / predict flow and add future prediction targets later.
