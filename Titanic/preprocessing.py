import os
import argparse
import joblib
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataTransformerInterface(ABC):
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class TitanicDataTransformer(DataTransformerInterface):
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).fillna(0)
        
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        X = df[features]
        y = df['Survived'] if 'Survived' in df.columns else None

        X_imputed = self.num_imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        processed_df = pd.DataFrame(X_scaled, columns=features)
        if y is not None:
            processed_df['target'] = y.values

        self.feature_names = features
        return processed_df

class ProcessingPipeline:
    def __init__(self, transformer: DataTransformerInterface):
        self.transformer = transformer

    def run(self, input_path: str, output_dir: str):
        df = pd.read_csv(input_path)
        processed_df = self.transformer.fit_transform(df)

        train_path = os.path.join(output_dir, "train")
        transformer_path = os.path.join(output_dir, "transformer")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(transformer_path, exist_ok=True)

        processed_df.to_csv(os.path.join(train_path, "train.csv"), index=False)
        joblib.dump(self.transformer, os.path.join(transformer_path, "transformer.joblib"))
        print("Preprocessing concluído e artefatos salvos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input/titanic.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    pipeline = ProcessingPipeline(transformer=TitanicDataTransformer())
    pipeline.run(args.input_data, args.output_dir)
