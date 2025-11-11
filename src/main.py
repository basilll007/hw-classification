from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.inference import predict_one, FEATURE_ORDER

app = FastAPI(title="Cancer Cell Classifier")

# Serve static assets (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve HTML templates
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CellInput(BaseModel):
    Gene_E_Housekeeping: float
    Gene_A_Oncogene: float
    Gene_B_Immune: float
    Gene_C_Stromal: float
    Gene_D_Therapy: float
    Pathway_Score_Inflam: float
    UMAP_1: float
    Disease_Status_Tumor: float | None = None
    Disease_Status: str | None = None

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(data: CellInput):
    result = predict_one(data.dict())
    return result
