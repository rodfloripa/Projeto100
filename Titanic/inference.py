import os
import json
import joblib
import pandas as pd
import shap
from io import StringIO

class InferenceService:
    def __init__(self, model_dir: str):
        self.model = joblib.load(os.path.join(model_dir, "model.joblib"))
        self.transformer = joblib.load(os.path.join(model_dir, "transformer.joblib"))
        self.explainer = shap.TreeExplainer(self.model)

    def predict_and_explain(self, raw_df: pd.DataFrame) -> dict:
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        raw_df['Sex'] = raw_df['Sex'].map({'male': 0, 'female': 1}).fillna(0)
        X = raw_df[features]
        
        X_imputed = self.transformer.num_imputer.transform(X)
        X_scaled = self.transformer.scaler.transform(X_imputed)

        predictions = self.model.predict(X_scaled)
        shap_values = self.explainer.shap_values(X_scaled)

        return {
            "predictions": predictions.tolist(),
            "explanations": {
                "shap_values": shap_values.tolist(),
                "base_value": float(self.explainer.expected_value)
            }
        }

def model_fn(model_dir: str) -> InferenceService:
    return InferenceService(model_dir)

def predict_fn(input_data, service: InferenceService) -> str:
    if isinstance(input_data, str):
        df = pd.read_csv(StringIO(input_data))
    else:
        df = pd.DataFrame(input_data)
        
    result = service.predict_and_explain(df)
    return json.dumps(result)
