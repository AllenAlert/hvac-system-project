import os
import json
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

# Suppress sklearn feature name warnings (we use numpy arrays, not DataFrames with names)
warnings.filterwarnings('ignore', message='X does not have valid feature names')

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
    
    
    def __init__(self, model, scaler=None, feature_names=None, model_type="random_forest"):
        self.model = model
        self.scaler = scaler
        self.model_type = model_type
        # Use provided feature names or default to the standard 8 features
        self.feature_names = feature_names or [
            'floor_area', 'height', 'window_ratio', 'insulation_r',
            'occupancy', 'outdoor_temp', 'solar_irradiance', 'wind_speed'
        ]
        # Default values for missing features
        self._defaults = {
            'floor_area': 1000.0,
            'height': 3.0,
            'window_ratio': 0.3,
            'insulation_r': 3.5,
            'occupancy': 20.0,
            'outdoor_temp': 20.0,
            'solar_irradiance': 500.0,
            'wind_speed': 5.0,
            'indoor_temp': 22.0,
            'setpoint': 22.0,
            'hour': 12.0,
        }
    
    def predict(self, df):
        """Predict energy consumption."""
        import numpy as np
        import pandas as pd
        # Handle both dict and DataFrame input
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        
        # Build feature array in correct order
        X = []
        for feat in self.feature_names:
            if feat in df.columns:
                X.append(float(df[feat].iloc[0]))
            else:
                X.append(self._defaults.get(feat, 0.0))
        
        X = np.array([X])
        
        # Apply scaler if available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance from the model."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(self.feature_names, np.abs(self.model.coef_)))
        return {name: 1.0 / len(self.feature_names) for name in self.feature_names}


class SimpleLoadPredictor:
    """Simple wrapper for sklearn load prediction model."""
    
    def __init__(self, model, scaler=None, feature_names=None, model_type="random_forest"):
        self.model = model
        self.scaler = scaler
        self.model_type = model_type
        # Use provided feature names or default to the standard 8 features
        self.feature_names = feature_names or [
            'floor_area', 'height', 'window_ratio', 'insulation_r',
            'occupancy', 'outdoor_temp', 'solar_irradiance', 'wind_speed'
        ]
        # Default values for missing features
        self._defaults = {
            'floor_area': 1000.0,
            'height': 3.0,
            'window_ratio': 0.3,
            'insulation_r': 3.5,
            'insulation_factor': 3.5,  # alias
            'occupancy': 20.0,
            'outdoor_temp': 20.0,
            'solar_irradiance': 500.0,
            'wind_speed': 5.0,
        }
    
    def predict(self, df):
        """Predict heating and cooling loads."""
        import numpy as np
        import pandas as pd
        # Handle both dict and DataFrame input
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        
        # Build feature array in correct order
        X = []
        for feat in self.feature_names:
            if feat in df.columns:
                X.append(float(df[feat].iloc[0]))
            elif feat == 'insulation_r' and 'insulation_factor' in df.columns:
                X.append(float(df['insulation_factor'].iloc[0]))
            else:
                X.append(self._defaults.get(feat, 0.0))
        
        X = np.array([X])
        
        # Apply scaler if available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        total_load = self.model.predict(X)
        
        # Get outdoor temp for heating/cooling split
        outdoor_temp = float(df['outdoor_temp'].iloc[0]) if 'outdoor_temp' in df.columns else 20.0
        
        # Rough heuristic for splitting heating vs cooling based on outdoor temp
        if outdoor_temp < 18:
            return total_load, np.array([0.0])  # heating season
        elif outdoor_temp > 24:
            return np.array([0.0]), total_load  # cooling season
        else:
            # shoulder season - split 50/50
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
                    # It's a dict with model, scaler, and metadata
                    self.energy_model = SimpleEnergyPredictor(
                        model=loaded['model'],
                        scaler=loaded.get('scaler'),
                        feature_names=loaded.get('feature_names'),
                        model_type=loaded.get('model_type', 'random_forest')
                    )
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
                    # It's a dict with model, scaler, and metadata
                    self.load_model = SimpleLoadPredictor(
                        model=loaded['model'],
                        scaler=loaded.get('scaler'),
                        feature_names=loaded.get('feature_names'),
                        model_type=loaded.get('model_type', 'random_forest')
                    )
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
            
            # Get feature importance (convert numpy types to Python floats)
            importance = self.energy_model.get_feature_importance()
            top_features = {
                k: float(v) for k, v in sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                )[:5]
            }
            
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
        if not ML_AVAILABLE:
            return {"error": "ML dependencies not available"}
        
        try:
            df = pd.DataFrame([building_params])
            outdoor_temp = building_params.get('outdoor_temp', 20.0)
            
            # Try load model first
            if self.load_model:
                try:
                    heating_pred, cooling_pred = self.load_model.predict(df)
                    return {
                        "heating_load": float(heating_pred[0]),
                        "cooling_load": float(cooling_pred[0]),
                        "total_load": float(heating_pred[0] + cooling_pred[0]),
                        "unit": "kW"
                    }
                except Exception:
                    pass  # Fall through to energy model fallback
            
            # Fallback: use energy model and split based on temperature
            if self.energy_model:
                total_energy = float(self.energy_model.predict(df)[0])
                # Split into heating/cooling based on outdoor temperature
                if outdoor_temp < 18:
                    # Heating dominant
                    heating_load = total_energy * 0.8
                    cooling_load = total_energy * 0.2
                elif outdoor_temp > 24:
                    # Cooling dominant
                    heating_load = total_energy * 0.1
                    cooling_load = total_energy * 0.9
                else:
                    # Shoulder season
                    heating_load = total_energy * 0.5
                    cooling_load = total_energy * 0.5
                
                return {
                    "heating_load": heating_load,
                    "cooling_load": cooling_load,
                    "total_load": heating_load + cooling_load,
                    "unit": "kW",
                    "note": "Estimated from energy model (load model unavailable)"
                }
            
            return {"error": "No prediction model available"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def optimize_setpoint(self, current_conditions: Dict[str, float], 
                         target_energy: float) -> Dict[str, Any]:
        """Optimize setpoint for target energy consumption."""
        if not ML_AVAILABLE or not self.energy_model:
            return {"error": "Model not available"}
        
        try:
            # Default parameters matching the model's expected features
            default_params = {
                'floor_area': 1000.0,
                'height': 3.0,
                'window_ratio': 0.3,
                'insulation_r': 3.5,
                'occupancy': 20.0,
                'outdoor_temp': 20.0,
                'solar_irradiance': 500.0,
                'wind_speed': 5.0,
            }
            
            # Merge with provided conditions (provided values override defaults)
            base_params = {**default_params, **current_conditions}
            
            # Test different setpoints (simulated by varying outdoor temp relationship)
            setpoints = np.arange(18, 26, 0.5)
            predictions = []
            
            for setpoint in setpoints:
                params = base_params.copy()
                # Simulate setpoint effect: lower setpoint = more cooling needed
                # This adjusts the effective "load" by modifying the temperature differential
                temp_differential = base_params.get('outdoor_temp', 20.0) - setpoint
                # Higher differential means more energy needed
                params['outdoor_temp'] = base_params.get('outdoor_temp', 20.0) + (setpoint - 22.0) * 0.5
                
                df = pd.DataFrame([params])
                pred = self.energy_model.predict(df)[0]
                predictions.append(float(pred))
            
            # Find closest to target
            predictions = np.array(predictions)
            best_idx = np.argmin(np.abs(predictions - target_energy))
            
            return {
                "optimal_setpoint": float(setpoints[best_idx]),
                "predicted_energy": float(predictions[best_idx]),
                "energy_savings": float(predictions[0] - predictions[best_idx]),
                "all_setpoints": [float(s) for s in setpoints],
                "all_predictions": [float(p) for p in predictions],
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