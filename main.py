from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "IRIS API is live!"}

@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])
    prediction = model.predict(features)
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return {
        "prediction": int(prediction[0]),
        "species": species_map[int(prediction[0])]
    }