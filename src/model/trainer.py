import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from preprocessing.preprocessor import DataPreprocessor
from features.feat_engineering import FeatureEngineer
from features.encoder import FeatureEncoder

class ModelTrainer:
    def __init__(self, train_path: str = "train.csv", model_save_path: str = "titanic_model.pkl"):
        self.train_path = train_path
        self.model_save_path = model_save_path
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)  
        self.pipeline = None

    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        data = pd.read_csv(self.train_path)
        X = data.drop(columns=['Survived'])
        y = data['Survived']
        return X, y

    def create_pipeline(self) -> Pipeline:
        return Pipeline([
            ('preprocessor', DataPreprocessor(encoder=self.encoder)),
            ('feature_engineer', FeatureEngineer()),
            ('encoder', FeatureEncoder(encoder=self.encoder)),
            ('classifier', self.model_gb)
        ])

    def train_with_cross_validation(self, cv: int = 5) -> float:
        X, y = self.load_data()
        
        self.pipeline = self.create_pipeline()

        cv_scores = cross_val_score(
            self.pipeline, 
            X, 
            y, 
            cv=cv, 
            scoring='accuracy'
        )
        mean_accuracy = cv_scores.mean()
        print(X.head())
        
        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, self.model_save_path)
        
        print(f"Mean acc on {cv}-fold cross-validation: {mean_accuracy:.4f}")
        return mean_accuracy