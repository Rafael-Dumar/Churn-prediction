import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


# example of a high risk customer
high_risk_customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 75.0,
    "TotalCharges": 150.0
}

# Define the data model for customer input
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        json_schema_extra = {
            "example": high_risk_customer
            
        }


# Create the FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn based on various features",
    version="1.0.0"
)

try:
    # load the pre-trained model and scaler
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    print("Model, columns and scaler loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("Model or scaler files not found. Run the training model first to generate them.")

@app.post("/predict_churn", tags=["Churn Prediction"])
async def predict_churn(customer_data: CustomerData):
    # Convert the input data to a dataframe
    input_df = pd.DataFrame([customer_data.model_dump()])

    # one-hot encode categorical variables
    input_encoded = pd.get_dummies(input_df)

    # ensure the input dataframe has the same columns as the model was trained on
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    # scale the input data
    input_scaled = scaler.transform(input_aligned)
    # convert to a dataframe
    input_scaled_df = pd.DataFrame(input_scaled, columns=model_columns)


    # make predictions
    churn_probability = model.predict_proba(input_scaled_df)[:, 1][0]
    churn_prediction = 'Yes' if churn_probability > 0.5 else 'No'

    return {
        "Churn Probability": float(f"{churn_probability:.4f}"),
        "Churn Prediction": churn_prediction
    }

from fastapi.responses import RedirectResponse
@app.get("/", tags=["Root"])
async def read_root():
    return RedirectResponse(url="/docs")
