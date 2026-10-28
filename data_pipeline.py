import os
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


class ChurnDataPipeline:
    """
    Modular data engineering pipeline for Telco Customer Churn.
    Demonstrates OOP, Type Hints, and strict prevention of Data Leakage.
    """

    def __init__(self, raw_data_path: str = "data/raw_churn.csv", output_dir: str = "data/processed") -> None:
        self.raw_data_path: str = raw_data_path
        self.output_dir: str = output_dir
        self.scaler: StandardScaler = StandardScaler()
        self.numeric_cols: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_data(self) -> pd.DataFrame:
        """Downloads raw dataset if not present locally and loads into a DataFrame."""
        if not os.path.exists(self.raw_data_path):
            print("Downloading Telco Customer Churn dataset...")
            df = pd.read_csv(DATA_URL)
            df.to_csv(self.raw_data_path, index=False)
        else:
            print("Loading cached raw dataset...")
            df = pd.read_csv(self.raw_data_path)
        return df

    def clean_and_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans corrupted numeric strings and encodes categorical variables."""
        df_clean = df.copy()

        # Drop unique identifier as it holds no predictive signal
        if "customerID" in df_clean.columns:
            df_clean.drop(columns=["customerID"], inplace=True)

        # TotalCharges has blank strings (" ") for new customers with tenure == 0
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

        # Convert target variable 'Churn' to binary 1/0
        df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})

        # One-hot encode categorical variables (drop_first=True avoids multicollinearity)
        categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
        df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True, dtype=int)

        return df_encoded

    def split_and_scale(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data BEFORE scaling to prevent data leakage.
        Fits StandardScaler strictly on training data.
        """
        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        # Stratified split preserves the ~26.5% churn class imbalance ratio in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Fit ONLY on X_train; transform both X_train and X_test
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[self.numeric_cols] = self.scaler.fit_transform(X_train[self.numeric_cols])
        X_test[self.numeric_cols] = self.scaler.transform(X_test[self.numeric_cols])

        # Save fitted scaler for production inference
        joblib.dump(self.scaler, os.path.join(self.output_dir, "scaler.pkl"))

        return X_train, X_test, y_train, y_test

    def run(self) -> None:
        """Executes end-to-end data pipeline and saves processed splits."""
        raw_df = self.fetch_data()
        encoded_df = self.clean_and_encode(raw_df)
        X_train, X_test, y_train, y_test = self.split_and_scale(encoded_df)

        X_train.to_csv(os.path.join(self.output_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.output_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.output_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.output_dir, "y_test.csv"), index=False)

        print(f"Pipeline complete! Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"Churn rate in Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")


if __name__ == "__main__":
    pipeline = ChurnDataPipeline()
    pipeline.run()