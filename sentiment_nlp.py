import pandas as pd
from transformers import pipeline

class CustomerIntelligenceEngine:
    """
    Combines structured Churn Prediction probabilities with
    unstructured NLP Sentiment Analysis using Hugging Face Transformers.
    """
    def __init__(self) -> None:
        print("Loading pre-trained NLP Transformer model...")
        # Using a lightweight, production-ready sentiment model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def get_mock_support_tickets(self) -> pd.DataFrame:
        """Simulates incoming customer support tickets and their ML churn probability."""
        return pd.DataFrame({
            "customer_id": ["CUST-001", "CUST-002", "CUST-003", "CUST-004"],
            "ml_churn_probability": [0.85, 0.62, 0.15, 0.78],
            "complaint_text": [
                "My internet keeps dropping every 5 minutes. Fix it or I'm leaving today!",
                "The connection is mostly fine, but my bill increased without any warning.",
                "Great customer support, the agent was very helpful with my router setup.",
                "I have been on hold for 45 minutes. This is completely unacceptable service."
            ]
        })

    def analyze_tickets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs Transformer NLP on text and applies business logic for retention."""
        print("Analyzing customer sentiment...")

        # Run text through the Hugging Face pipeline
        texts = df["complaint_text"].tolist()
        nlp_results = self.sentiment_analyzer(texts)

        # Extract labels (POSITIVE/NEGATIVE) and confidence scores
        df["sentiment_label"] = [res["label"] for res in nlp_results]
        df["sentiment_confidence"] = [round(res["score"], 4) for res in nlp_results]

        # Business Logic Engine:
        # If ML predicts high churn (>60%) AND the customer is actively angry -> URGENT
        df["action_required"] = df.apply(
            lambda row: "URGENT_RETENTION_CALL"
            if row["ml_churn_probability"] > 0.60 and row["sentiment_label"] == "NEGATIVE"
            else "STANDARD_MONITORING",
            axis=1
        )

        return df

if __name__ == "__main__":
    engine = CustomerIntelligenceEngine()

    raw_tickets = engine.get_mock_support_tickets()
    processed_tickets = engine.analyze_tickets(raw_tickets)

    print("\n--- FINAL CUSTOMER INTELLIGENCE DASHBOARD ---")
    # Displaying specific columns for a clean terminal output
    print(processed_tickets[["customer_id", "ml_churn_probability", "sentiment_label", "action_required"]].to_string(index=False))