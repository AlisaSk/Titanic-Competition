from sklearn.base import BaseEstimator, TransformerMixin

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, encoder):
        self.encoder = encoder
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data['Age'] = data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
        data["Fare"] = data["Fare"].fillna(data["Fare"].median())
        
        return data