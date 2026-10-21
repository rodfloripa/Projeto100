import os
import argparse
import joblib
import json
import hashlib
import pandas as pd
from xgboost import XGBClassifier

class ReproducibleTrainer:
    def __init__(self, seed: int, max_depth: int, n_estimators: int):
        self.seed = seed
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.model = XGBClassifier(
            random_state=self.seed, 
            max_depth=self.max_depth, 
            n_estimators=self.n_estimators
        )

    def generate_config_hash(self) -> str:
        config = {
            "seed": self.seed,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators
        }
        return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def train(self, data_path: str, model_dir: str):
        df = pd.read_csv(os.path.join(data_path, "train.csv"))
        X = df.drop("target", axis=1)
        y = df["target"]

        print(f"Treinando modelo com Config Hash: {self.generate_config_hash()}")
        self.model.fit(X, y)

        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "model.joblib"))
        print("Modelo salvo com sucesso.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--train-data", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()

    trainer = ReproducibleTrainer(
        seed=args.seed, 
        max_depth=args.max_depth, 
        n_estimators=args.n_estimators
    )
    trainer.train(args.train_data, args.model_dir)
