from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model and Label Encoder
model = joblib.load("disease_prediction_model.pkl")
le = joblib.load("label_encoder.pkl")

@app.post("/predict/")
def predict(symptoms: list):
    new_data = pd.DataFrame([symptoms], columns=["symptom1", "symptom2", "symptom3"])
    prediction = model.predict(new_data)[0]
    predicted_disease = le.inverse_transform([prediction])[0]
    return {"Predicted Disease": predicted_disease}

@app.get("/")
def home():
    return {"message": "FastAPI Disease Prediction Model Running"}