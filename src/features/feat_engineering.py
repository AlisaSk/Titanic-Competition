import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config import Config

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data['AgeGroup'] = pd.cut(data['Age'], 
                                  bins=Config.AGE_BINS, 
                                  labels=Config.AGE_LABELS).astype(int)
        data["FamilySize"] = data['SibSp'] + data['Parch'] + 1
        data["Cabin"] = data["Cabin"].str[0]
        
        return data