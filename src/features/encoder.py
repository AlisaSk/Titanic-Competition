import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config import Config

class FeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder):
        self.encoder = encoder
        
    def fit(self, X, y=None):
        self.encoder.fit(X[Config.FEATURES_TO_ENCODE])
        return self
    
    def transform(self, X):
        data = X.copy()
        encoded_features = self.encoder.transform(data[Config.FEATURES_TO_ENCODE])
        encoded_df = pd.DataFrame(
            encoded_features, 
            columns=self.encoder.get_feature_names_out(Config.FEATURES_TO_ENCODE),
            index=data.index
        )
        
        data = data.drop(columns=Config.FEATURES_TO_ENCODE)
        
        data = pd.concat([data, encoded_df], axis=1)
        
        data = data.drop(columns=Config.FEATURES_TO_DROP)
        
        return data
