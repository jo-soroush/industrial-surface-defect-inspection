from fastapi import FastAPI

from api.app.routes.predict import router as predict_router
from api.app.routes.system import router as system_router


app = FastAPI(title="Industrial Surface Defect Inspection Platform API")

app.include_router(system_router)
app.include_router(predict_router)
