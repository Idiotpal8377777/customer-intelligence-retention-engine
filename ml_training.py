import os
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib


class ModelTrainer:
    """
    Trains, benchmarks, and exports classical ML models on processed churn data.
    Evaluates Logistic Regression, Random Forest, and XGBoost with a focus on Recall.
    """

    def __init__(self, data_dir: str = "data/processed", model_dir: str = "models") -> None:
        self.data_dir: str = data_dir
        self.model_dir: str = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.X_train: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, "X_train.csv"))
        self.X_test: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, "X_test.csv"))
        self.y_train: pd.Series = pd.read_csv(os.path.join(self.data_dir, "y_train.csv")).squeeze()
        self.y_test: pd.Series = pd.read_csv(os.path.join(self.data_dir, "y_test.csv")).squeeze()

        # Class weight compensation: ~73.5% non-churn vs ~26.5% churn
        self.scale_pos_weight = (len(self.y_train) - sum(self.y_train)) / sum(self.y_train)

    def get_models(self) -> Dict[str, Any]:
        """Returns initialized candidate models with class weighting to penalize missed churners."""
        return {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced",
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=4,
                scale_pos_weight=self.scale_pos_weight,
                eval_metric="logloss",
                random_state=42
            )
        }

    def train_and_evaluate(self) -> pd.DataFrame:
        """Trains each model and returns a comparative performance leaderboard."""
        models = self.get_models()
        results = []
        best_recall = -1.0
        best_model_name = ""
        best_model_obj = None

        print("\nStarting Model Training & Evaluation...\n" + "=" * 50)

        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]

            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_proba)
            cm = confusion_matrix(self.y_test, y_pred)

            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall (Priority)": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(auc, 4)
            })

            print(f"[{name}]")
            print("Confusion_Matrix")
            print(cm)
            print(f"Recall: {rec:.2%} | ROC-AUC: {auc:.4f}\n")

            # Selection logic: Prioritizing Recall while maintaining reasonable ROC-AUC
            if rec > best_recall:
                best_recall = rec
                best_model_name = name
                best_model_obj = model

        leaderboard = pd.DataFrame(results).sort_values(by="Recall (Priority)", ascending=False)

        # Export best model artifact for deployment
        best_model_path = os.path.join(self.model_dir, "best_model.pkl")
        joblib.dump(best_model_obj, best_model_path)
        print("=" * 50)
        print(f"Selected [{best_model_name}] as production model based on highest Recall ({best_recall:.2%}).")
        print(f"Model saved to: {best_model_path}")

        return leaderboard


if __name__ == "__main__":
    trainer = ModelTrainer()
    df_results = trainer.train_and_evaluate()
    print("\nFinal Model Leaderboard:\n")
    print(df_results.to_string(index=False))