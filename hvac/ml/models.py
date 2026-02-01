import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
from typing import Dict, List, Tuple, Optional, Any

class BasePredictor:
    """Base class for HVAC prediction models."""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def _create_model(self):
        """Create the underlying ML model."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        self.feature_names = list(X.columns)
        self.model = self._create_model()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']

class EnergyPredictor(BasePredictor):
    """Predict total energy consumption of buildings."""
    
    def __init__(self, model_type: str = 'xgboost'):
        super().__init__(model_type)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            return {}
            
        return dict(zip(self.feature_names, importance))

class LoadPredictor(BasePredictor):
    """Predict heating and cooling loads separately."""
    
    def __init__(self, model_type: str = 'random_forest'):
        super().__init__(model_type)
        self.heating_model = None
        self.cooling_model = None
        
    def fit(self, X: pd.DataFrame, y_heating: pd.Series, y_cooling: pd.Series):
        """Train separate models for heating and cooling."""
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train heating model
        self.heating_model = self._create_model()
        self.heating_model.fit(X_scaled, y_heating)
        
        # Train cooling model  
        self.cooling_model = self._create_model()
        self.cooling_model.fit(X_scaled, y_cooling)
        
        self.is_trained = True
        
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict both heating and cooling loads."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X_scaled = self.scaler.transform(X)
        heating_pred = self.heating_model.predict(X_scaled)
        cooling_pred = self.cooling_model.predict(X_scaled)
        
        return heating_pred, cooling_pred
    
    def predict_total(self, X: pd.DataFrame) -> np.ndarray:
        """Predict total load (heating + cooling)."""
        heating, cooling = self.predict(X)
        return heating + cooling

class TimeSeriesPredictor:
    """Predict energy consumption time series."""
    
    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours
        self.model = xgb.XGBRegressor(n_estimators=200, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_sequences(self, data: pd.DataFrame, 
                        target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        X, y = [], []
        
        for i in range(self.lookback_hours, len(data)):
            # Features: past values + current weather
            sequence = data.iloc[i-self.lookback_hours:i][target_col].values
            current_weather = data.iloc[i][['outdoor_temp', 'solar_irradiance']].values
            features = np.concatenate([sequence, current_weather])
            
            X.append(features)
            y.append(data.iloc[i][target_col])
            
        return np.array(X), np.array(y)
    
    def fit(self, data: pd.DataFrame, target_col: str = 'total_energy'):
        """Train the time series model."""
        X, y = self.create_sequences(data, target_col)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
    def predict_next(self, recent_data: pd.DataFrame, 
                    target_col: str = 'total_energy') -> float:
        """Predict next time step."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        if len(recent_data) < self.lookback_hours + 1:
            raise ValueError(f"Need at least {self.lookback_hours + 1} data points")
            
        # Create sequence
        sequence = recent_data.iloc[-self.lookback_hours-1:-1][target_col].values
        current_weather = recent_data.iloc[-1][['outdoor_temp', 'solar_irradiance']].values
        features = np.concatenate([sequence, current_weather]).reshape(1, -1)
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)[0]