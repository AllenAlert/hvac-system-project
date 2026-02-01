import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from .models import EnergyPredictor, LoadPredictor, TimeSeriesPredictor
from .data_generator import HVACDataGenerator
from .utils import prepare_features, evaluate_model

class ModelTrainer:
    """Handle model training workflows and evaluation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}
        
    def train_energy_model(self, data: pd.DataFrame, 
                          target_col: str = 'total_energy',
                          model_type: str = 'xgboost',
                          test_size: float = 0.2) -> EnergyPredictor:
        """Train energy prediction model."""
        
        # Prepare features
        feature_cols = [col for col in data.columns if col not in 
                       ['heating_load', 'cooling_load', 'total_energy']]
        X = data[feature_cols]
        y = data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Train model
        model = EnergyPredictor(model_type=model_type)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        
        self.results[f'energy_{model_type}'] = {
            'model': model,
            'metrics': metrics,
            'feature_importance': model.get_feature_importance()
        }
        
        print(f"Energy Model ({model_type}) Performance:")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        
        return model
    
    def train_load_model(self, data: pd.DataFrame,
                        model_type: str = 'random_forest',
                        test_size: float = 0.2) -> LoadPredictor:
        """Train separate heating and cooling load models."""
        
        # Prepare features
        feature_cols = [col for col in data.columns if col not in 
                       ['heating_load', 'cooling_load', 'total_energy']]
        X = data[feature_cols]
        y_heating = data['heating_load']
        y_cooling = data['cooling_load']
        
        # Split data
        X_train, X_test, y_h_train, y_h_test, y_c_train, y_c_test = train_test_split(
            X, y_heating, y_cooling, test_size=test_size, random_state=self.random_state
        )
        
        # Train model
        model = LoadPredictor(model_type=model_type)
        model.fit(X_train, y_h_train, y_c_train)
        
        # Evaluate
        y_h_pred, y_c_pred = model.predict(X_test)
        
        heating_metrics = evaluate_model(y_h_test, y_h_pred)
        cooling_metrics = evaluate_model(y_c_test, y_c_pred)
        
        self.results[f'load_{model_type}'] = {
            'model': model,
            'heating_metrics': heating_metrics,
            'cooling_metrics': cooling_metrics
        }
        
        print(f"Load Model ({model_type}) Performance:")
        print(f"Heating - R²: {heating_metrics['r2']:.4f}, RMSE: {heating_metrics['rmse']:.2f}")
        print(f"Cooling - R²: {cooling_metrics['r2']:.4f}, RMSE: {cooling_metrics['rmse']:.2f}")
        
        return model
    
    def train_time_series_model(self, data: pd.DataFrame,
                               target_col: str = 'total_energy',
                               lookback_hours: int = 24) -> TimeSeriesPredictor:
        """Train time series prediction model."""
        
        # Split data (use last 20% for testing)
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        # Train model
        model = TimeSeriesPredictor(lookback_hours=lookback_hours)
        model.fit(train_data, target_col)
        
        # Evaluate on test set
        predictions = []
        actuals = []
        
        for i in range(lookback_hours + 1, len(test_data)):
            recent_data = pd.concat([
                train_data.tail(lookback_hours),
                test_data[:i]
            ])
            
            pred = model.predict_next(recent_data, target_col)
            actual = test_data.iloc[i][target_col]
            
            predictions.append(pred)
            actuals.append(actual)
            
        metrics = evaluate_model(np.array(actuals), np.array(predictions))
        
        self.results[f'timeseries_{target_col}'] = {
            'model': model,
            'metrics': metrics
        }
        
        print(f"Time Series Model Performance:")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        
        return model
    
    def hyperparameter_tuning(self, data: pd.DataFrame,
                             target_col: str = 'total_energy',
                             model_type: str = 'xgboost') -> Dict[str, Any]:
        """Perform hyperparameter tuning."""
        
        feature_cols = [col for col in data.columns if col not in 
                       ['heating_load', 'cooling_load', 'total_energy']]
        X = data[feature_cols]
        y = data[target_col]
        
        if model_type == 'xgboost':
            from xgboost import XGBRegressor
            model = XGBRegressor(random_state=self.random_state)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=self.random_state)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        else:
            raise ValueError(f"Hyperparameter tuning not implemented for {model_type}")
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='r2', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"Best parameters for {model_type}:")
        print(results['best_params'])
        print(f"Best CV score: {results['best_score']:.4f}")
        
        return results
    
    def compare_models(self, data: pd.DataFrame,
                      target_col: str = 'total_energy',
                      models: List[str] = None) -> pd.DataFrame:
        """Compare performance of different models."""
        
        if models is None:
            models = ['linear', 'random_forest', 'xgboost']
        
        results = []
        
        for model_type in models:
            print(f"\nTraining {model_type} model...")
            model = self.train_energy_model(data, target_col, model_type)
            
            result = self.results[f'energy_{model_type}']['metrics'].copy()
            result['model_type'] = model_type
            results.append(result)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('r2', ascending=False)
        
        print("\nModel Comparison:")
        print(comparison_df[['model_type', 'r2', 'rmse', 'mae']])
        
        return comparison_df
    
    def plot_predictions(self, model_name: str, data: pd.DataFrame,
                        target_col: str = 'total_energy', n_samples: int = 100):
        """Plot actual vs predicted values."""
        
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        model = self.results[model_name]['model']
        
        # Get test predictions
        feature_cols = [col for col in data.columns if col not in 
                       ['heating_load', 'cooling_load', 'total_energy']]
        X = data[feature_cols].sample(n_samples, random_state=self.random_state)
        y_actual = data.loc[X.index, target_col]
        
        if isinstance(model, LoadPredictor):
            y_pred = model.predict_total(X)
        else:
            y_pred = model.predict(X)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_actual, y_pred, alpha=0.6)
        plt.plot([y_actual.min(), y_actual.max()], 
                [y_actual.min(), y_actual.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return y_actual, y_pred