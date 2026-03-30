from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from lime import lime_tabular

app = FastAPI()

# --- 1. Models and Data Loading ---
model_det = joblib.load("det_model.pkl")
model_cls = joblib.load("cls_model.pkl")
scaler_det = joblib.load("scaler_det.pkl")
scaler_cls = joblib.load("scaler_cls.pkl")
cols = list(joblib.load("feature_names.pkl"))
train_sample = joblib.load("training_sample.pkl")

# --- 2. Request Schema ---
class DiagnosisRequest(BaseModel):
    Gender: float
    Age: float
    BSR: float
    Systolic: float
    Diastolic: float
    BMI: float
    Peripheral_Neuropathy: float
    Delayed_Healing: float
    Genetic_Relation: float
    Frequent_Urination: float
    Dry_Mouth: float
    Frequent_Hunger: float
    role: str  # "patient" ya "doctor"

# --- 3. LIME Setup ---
explainer = lime_tabular.LimeTabularExplainer(
    training_data=train_sample,
    feature_names=cols,
    class_names=["Non-Diabetic", "Diabetic"],
    mode="classification"
)

@app.get("/")
def home():
    return {"message": "Diabetes API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_diabetes(request: DiagnosisRequest):
    try:
        data_dict = request.dict()
        role = data_dict.pop("role")

        # Convert Data in DataFrame
        input_df = pd.DataFrame([data_dict], columns=cols)

        # --- STAGE 1: DETECTION ---
        input_scaled_det = scaler_det.transform(input_df)
        pred_det = model_det.predict(input_scaled_det)[0]
        prob_det = model_det.predict_proba(input_scaled_det)[0]
        confidence_det = round(float(max(prob_det) * 100), 2)

        result = {}

        if pred_det == 0:
            result = {
                "status": "Non-Diabetic",
                "stage": 1,
                "stage1_confidence": confidence_det
            }
        else:
            # --- STAGE 2: CLASSIFICATION ---
            input_scaled_cls = scaler_cls.transform(input_df)
            pred_cls = model_cls.predict(input_scaled_cls)[0]
            prob_cls = model_cls.predict_proba(input_scaled_cls)[0]
            confidence_cls = round(float(max(prob_cls) * 100), 2)

            type_label = str(pred_cls).capitalize()

            result = {
                "status": "Diabetic Detected",
                "type": type_label,
                "stage": 2,
                "stage1_confidence": confidence_det,
                "stage2_confidence": confidence_cls
            }

        # --- 4. XAI (Only for Doctor) ---
        if role.lower() == "doctor":
            exp = explainer.explain_instance(
                input_scaled_det[0],
                model_det.predict_proba,
                num_features=len(cols)
            )

            result["xai_data"] = [
                {"feature": feature, "weight": float(weight)}
                for feature, weight in exp.as_list()
            ]

        return result

    except Exception as e:
        return {"error": str(e)}
