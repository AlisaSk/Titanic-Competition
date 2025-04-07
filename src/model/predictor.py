import pandas as pd
import joblib

class TitanicPredictor:
    def __init__(self, model_path: str = "titanic_model.pkl"):
        self.model = joblib.load(model_path)

    def load_test_data(self, test_path: str = "test.csv") -> pd.DataFrame:
        return pd.read_csv(test_path)

    def predict(self, test_data: pd.DataFrame) -> pd.Series:
        return pd.Series(
            self.model.predict(test_data),
            name="Survived"
        )

    def save_predictions(self, predictions: pd.Series, 
                        test_data: pd.DataFrame,
                        output_path: str = "predictions.csv") -> None:
        submission = pd.DataFrame({
            "PassengerId": test_data["PassengerId"],
            "Survived": predictions
        })
        submission.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    def run(self, test_path: str = "test.csv", 
           output_path: str = "predictions.csv") -> pd.DataFrame:
        
        test_data = self.load_test_data(test_path)
        predictions = self.predict(test_data)
        self.save_predictions(predictions, test_data, output_path)
        
        return pd.DataFrame({
            "PassengerId": test_data["PassengerId"],
            "Survived": predictions
        })