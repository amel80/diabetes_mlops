from fastapi import FastAPI, HTTPException,Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi.responses import HTMLResponse
app = FastAPI()
# Configuration du dossier de templates
templates = Jinja2Templates(directory="templates")

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
dict_res = {0: 'Not-Diabetes', 1: 'Diabetes'}

# pipeline_path = '/app/models/pipeline.pkl'
# with open(pipeline_path, 'rb') as pipeline_file:
#     pipeline = pickle.load(pipeline_file)
# Fonction pour charger le meilleur modèle depuis MLflow
# Charger le meilleur modèle à partir de MLflow
def load_best_model_from_mlflow(experiment_name: str):
    try:
        # Obtenir l'ID de l'expérimentation
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        # Rechercher le meilleur run basé sur le f1_score
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        best_run = runs.loc[runs['metrics.f1_score'].idxmax()]
        
        print(f"Best Run ID: {best_run.run_id}")
        
        # Charger le modèle avec l'URI
        best_model_uri = f"runs:/{best_run.run_id}/model"
        best_model = mlflow.sklearn.load_model(best_model_uri)

        return best_model
    except Exception as e:
        print("Error loading best model:", str(e))
        raise Exception("Unable to load model from MLflow.")

# Nom de l'expérimentation
experiment_name = "nom_de_votre_experiment"

# Charger le modèle au démarrage de l'application
try:
    pipeline = load_best_model_from_mlflow(experiment_name)
except Exception as e:
    print(e)
    pipeline = None
    
# Schéma d'entrée des données
class DataInput(BaseModel):
    data: list

@app.post("/predict")
async def predict(input_data: DataInput):
    try:
        df = pd.DataFrame(input_data.data, columns=columns)
        predictions = pipeline.predict(df)
        results = [dict_res[pred] for pred in predictions]
    
        return {"predictions": results}
    
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

# Route pour la page principale
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
