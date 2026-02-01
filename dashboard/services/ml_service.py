"""
ML Service - energy and load prediction

Added ML predictions to make the dashboard more useful.
The models are trained on simulated data but work pretty well
for quick estimates.

@author: Bola
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

# the hvac.ml module is optional - don't want to force sklearn on everyone
try:
    from hvac.ml import HVACDataGenerator, ModelTrainer, EnergyPredictor, LoadPredictor
    from hvac.ml.models import TimeSeriesPredictor
    from hvac.ml.utils import prepare_features, evaluate_model
    ML_AVAILABLE = True
except ImportError:
    # no sklearn/xgboost installed, thats fine
    ML_AVAILABLE = False


class SimpleEnergyPredictor:
    """
    Wrapper for the sklearn model. Had to add this because the hvac.ml
    predictors have a different interface and I didn't want to refactor everything.
    """
    
    def __init__(self, model):
        self.model = model
        self.model_type = "random_forest"  # or xgboost, doesn't really matter here
        # IMPORTANT: order must match training data!!
        self.feature_names = ['outdoor_temp', 'indoor_temp', 'setpoint', 'occupancy', 'hour']
    
    def predict(self, df):
        """Predict energy consumption."""
        import numpy as np
        import pandas as pd
        # Handle both dict and DataFrame input
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        # Extract features in correct order as numpy array (no feature names)
        X = np.array([[
            float(df['outdoor_temp'].iloc[0]) if 'outdoor_temp' in df.columns else float(df.get('outdoor_temperature', [20]).iloc[0]) if 'outdoor_temperature' in df.columns else 20.0,
            float(df['indoor_temp'].iloc[0]) if 'indoor_temp' in df.columns else float(df.get('indoor_temperature', [22]).iloc[0]) if 'indoor_temperature' in df.columns else 22.0,
            float(df['setpoint'].iloc[0]) if 'setpoint' in df.columns else 22.0,
            float(df['occupancy'].iloc[0]) if 'occupancy' in df.columns else 1.0,
            float(df['hour'].iloc[0]) if 'hour' in df.columns else float(df.get('hour_of_day', [12]).iloc[0]) if 'hour_of_day' in df.columns else 12.0
        ]])
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance from the model."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {name: 0.2 for name in self.feature_names}


class SimpleLoadPredictor:
    """Simple wrapper for sklearn load prediction model."""
    
    def __init__(self, model):
        self.model = model
        self.feature_names = ['outdoor_temp', 'floor_area', 'insulation_factor', 'window_ratio']
    
    def predict(self, df):
        """Predict heating and cooling loads."""
        import numpy as np
        import pandas as pd
        # Handle both dict and DataFrame input
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        
        outdoor_temp = float(df['outdoor_temp'].iloc[0]) if 'outdoor_temp' in df.columns else 20.0
        floor_area = float(df['floor_area'].iloc[0]) if 'floor_area' in df.columns else float(df.get('building_area', [100]).iloc[0]) if 'building_area' in df.columns else 100.0
        insulation = float(df['insulation_factor'].iloc[0]) if 'insulation_factor' in df.columns else 1.0
        window_ratio = float(df['window_ratio'].iloc[0]) if 'window_ratio' in df.columns else 0.2
        
        X = np.array([[outdoor_temp, floor_area, insulation, window_ratio]])
        total_load = self.model.predict(X)
        
        # super rough heuristic for splitting heating vs cooling
        # TODO: this should really be based on the balance point temp, not hardcoded 18/24
        if outdoor_temp < 18:
            return total_load, np.array([0.0])  # heating season
        elif outdoor_temp > 24:
            return np.array([0.0]), total_load  # cooling season
        else:
            # shoulder season - just split 50/50, close enough
            return total_load * 0.5, total_load * 0.5


class MLService:
    """ML service for HVAC dashboard."""
    
    def __init__(self, models_dir: str = "dashboard/data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.energy_model = None
        self.load_model = None
        self.ts_model = None
        self.data_generator = None
        
        if ML_AVAILABLE:
            self.data_generator = HVACDataGenerator()
            self._load_models()
    
    def is_available(self) -> bool:
        """Check if ML functionality is available."""
        return ML_AVAILABLE
    
    def _load_models(self):
        """Load pre-trained models if they exist."""
        import joblib
        try:
            energy_path = self.models_dir / "energy_model.joblib"
            if energy_path.exists():
                loaded = joblib.load(str(energy_path))
                # Check if it's already an EnergyPredictor or a raw model
                if hasattr(loaded, 'predict') and hasattr(loaded, 'get_feature_importance'):
                    # It's already an EnergyPredictor from hvac.ml
                    self.energy_model = loaded
                elif isinstance(loaded, dict) and 'model' in loaded:
                    # It's a dict with model and metadata
                    self.energy_model = SimpleEnergyPredictor(loaded['model'])
                else:
                    # It's a raw sklearn model
                    self.energy_model = SimpleEnergyPredictor(loaded)
                
            load_path = self.models_dir / "load_model.joblib"
            if load_path.exists():
                loaded = joblib.load(str(load_path))
                # Check if it's already a LoadPredictor or a raw model
                if hasattr(loaded, 'predict') and hasattr(loaded, 'heating_model'):
                    # It's already a LoadPredictor from hvac.ml
                    self.load_model = loaded
                elif isinstance(loaded, dict) and 'model' in loaded:
                    # It's a dict with model and metadata
                    self.load_model = SimpleLoadPredictor(loaded['model'])
                else:
                    # It's a raw sklearn model
                    self.load_model = SimpleLoadPredictor(loaded)
        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            pass  # Models will be trained on demand
    
    def train_models(self, n_samples: int = 1000) -> Dict[str, Any]:
        """Train ML models with generated data."""
        if not ML_AVAILABLE:
            return {"error": "ML dependencies not available"}
        
        try:
            # Generate training data
            data = self.data_generator.generate_building_dataset(n_samples)
            
            # Train models
            trainer = ModelTrainer()
            
            # Energy model
            self.energy_model = trainer.train_energy_model(
                data, model_type='xgboost', test_size=0.2
            )
            
            # Load model
            self.load_model = trainer.train_load_model(
                data, model_type='random_forest', test_size=0.2
            )
            
            # Save models
            self.energy_model.save(str(self.models_dir / "energy_model.joblib"))
            self.load_model.save(str(self.models_dir / "load_model.joblib"))
            
            # Get performance metrics
            energy_metrics = trainer.results[f'energy_xgboost']['metrics']
            load_metrics = trainer.results[f'load_random_forest']
            
            return {
                "success": True,
                "samples_trained": n_samples,
                "energy_model": {
                    "r2": energy_metrics['r2'],
                    "rmse": energy_metrics['rmse'],
                    "mae": energy_metrics['mae']
                },
                "load_model": {
                    "heating_r2": load_metrics['heating_metrics']['r2'],
                    "cooling_r2": load_metrics['cooling_metrics']['r2']
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def predict_energy(self, building_params: Dict[str, float]) -> Dict[str, Any]:
        """Predict energy consumption for given building parameters."""
        if not ML_AVAILABLE or not self.energy_model:
            return {"error": "Energy model not available"}
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([building_params])
            
            # Make prediction
            prediction = self.energy_model.predict(df)[0]
            
            # Get feature importance
            importance = self.energy_model.get_feature_importance()
            top_features = dict(sorted(importance.items(), 
                                     key=lambda x: x[1], reverse=True)[:5])
            
            return {
                "predicted_energy": float(prediction),
                "unit": "kW",
                "top_features": top_features,
                "model_type": self.energy_model.model_type
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def predict_loads(self, building_params: Dict[str, float]) -> Dict[str, Any]:
        """Predict heating and cooling loads."""
        if not ML_AVAILABLE or not self.load_model:
            return {"error": "Load model not available"}
        
        try:
            df = pd.DataFrame([building_params])
            heating_pred, cooling_pred = self.load_model.predict(df)
            
            return {
                "heating_load": float(heating_pred[0]),
                "cooling_load": float(cooling_pred[0]),
                "total_load": float(heating_pred[0] + cooling_pred[0]),
                "unit": "kW"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def optimize_setpoint(self, current_conditions: Dict[str, float], 
                         target_energy: float) -> Dict[str, Any]:
        """Optimize setpoint for target energy consumption."""
        if not ML_AVAILABLE or not self.energy_model:
            return {"error": "Model not available"}
        
        try:
            # Test different setpoints
            setpoints = np.arange(18, 26, 0.5)
            predictions = []
            
            for setpoint in setpoints:
                params = current_conditions.copy()
                # Adjust parameters based on setpoint
                params['outdoor_temp'] = current_conditions.get('outdoor_temp', 20)
                
                df = pd.DataFrame([params])
                pred = self.energy_model.predict(df)[0]
                predictions.append(pred)
            
            # Find closest to target
            predictions = np.array(predictions)
            best_idx = np.argmin(np.abs(predictions - target_energy))
            
            return {
                "optimal_setpoint": float(setpoints[best_idx]),
                "predicted_energy": float(predictions[best_idx]),
                "energy_savings": float(predictions[0] - predictions[best_idx]),
                "unit": "kW"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of ML models."""
        return {
            "ml_available": ML_AVAILABLE,
            "energy_model_loaded": self.energy_model is not None,
            "load_model_loaded": self.load_model is not None,
            "models_directory": str(self.models_dir)
        }
    
    def generate_forecast(self, history_data: List[Dict], 
                         hours_ahead: int = 24) -> Dict[str, Any]:
        """Generate energy consumption forecast."""
        if not ML_AVAILABLE or len(history_data) < 24:
            return {"error": "Insufficient data or ML not available"}
        
        try:
            # Convert history to DataFrame
            df = pd.DataFrame(history_data)
            
            # Simple forecast using recent trends
            recent_energy = [d.get('cooling', 0) for d in history_data[-24:]]
            avg_energy = np.mean(recent_energy)
            trend = np.polyfit(range(len(recent_energy)), recent_energy, 1)[0]
            
            forecast = []
            for h in range(hours_ahead):
                predicted = avg_energy + trend * h
                forecast.append(max(0, predicted))
            
            return {
                "forecast": forecast,
                "hours_ahead": hours_ahead,
                "average_recent": float(avg_energy),
                "trend": float(trend),
                "unit": "W"
            }
            
        except Exception as e:
            return {"error": str(e)}

# Global ML service instance
_ml_service = None

def get_ml_service() -> MLService:
    """Get the global ML service instance."""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service