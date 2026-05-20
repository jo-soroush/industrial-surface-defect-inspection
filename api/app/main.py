from fastapi import FastAPI

from api.app.routes.system import router as system_router


app = FastAPI(title="Industrial Surface Defect Inspection Platform API")

app.include_router(system_router)
