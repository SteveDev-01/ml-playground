from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('models/best_model.pkl')

@app.post('/predict')
def predict(payload: dict):
    df = pd.DataFrame([payload])
    prediction = model.predict(df)[0]
    return {'prediction': int(prediction)}
