# Frontend Dashboard

This directory contains the Streamlit frontend dashboard for the Industrial Surface Defect Inspection Platform.

## Run Locally

Install the project requirements first:

```bash
python -m pip install -r requirements.txt
```

That install also includes Plotly for cleaner dashboard charts.

Start the API in one terminal:

```bash
uvicorn api.app.main:app --reload --port 8000
```

Then start the Streamlit app in another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

## What This Dashboard Does

- Shows the project title and current high-level status
- Loads and validates the existing JSON data contracts for the surface defect, surface anomaly, and defect detection evidence bundles
- Provides sidebar navigation for the planned dashboard pages
- Presents a safe limitations banner
- Includes a local image inspection page that currently calls the classification endpoint
- Establishes the structure for reading existing JSON data contracts in the next phase

## What Is Not Implemented Yet

- Unified image inspection wiring in the Streamlit page
- Docker or docker-compose
- Full chart wiring from JSON data contracts
- Production or deployment claims

## Data Loading Phase

The current frontend phase validates that the following JSON data-contract folders are present and readable:

- `artifacts/frontend/track_a/`
- `artifacts/frontend/track_b/`
- `artifacts/frontend/detection/yolo_train_v0_2_0/`

The next frontend phase is to refine the local inspection workflow and turn these validated payloads into dashboard sections and charts.

The local inspection page expects the API to be available at `http://localhost:8000` by default.

## Next Phase

The next frontend phase is to refine the local image inspection flow and connect the unified inspection response.
