import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Any

def prepare_features(data: pd.DataFrame, 
                    target_cols: List[str] = None,
                    scale_features: bool = True,
                    add_derived_features: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare features for ML models."""
    
    if target_cols is None:
        target_cols = ['heating_load', 'cooling_load', 'total_energy']
    
    # Separate features and targets
    feature_cols = [col for col in data.columns if col not in target_cols]
    X = data[feature_cols].copy()
    y = data[target_cols].copy() if len(target_cols) > 1 else data[target_cols[0]].copy()
    
    # Add derived features
    if add_derived_features:
        X = add_engineering_features(X)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Scale features
    if scale_features:
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y

def add_engineering_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add engineering-derived features."""
    df = data.copy()
    
    # Building envelope features
    if 'floor_area' in df.columns and 'height' in df.columns:
        df['volume'] = df['floor_area'] * df['height']
        df['surface_area'] = df['floor_area'] * 2 + df['floor_area'] * 0.4 * 4  # Simplified
    
    if 'window_ratio' in df.columns:
        df['wall_ratio'] = 1 - df['window_ratio']
    
    # Thermal features
    if 'insulation_r' in df.columns:
        df['u_value'] = 1.0 / df['insulation_r']
    
    # Weather features
    if 'outdoor_temp' in df.columns:
        df['heating_degree_days'] = np.maximum(0, 18 - df['outdoor_temp'])
        df['cooling_degree_days'] = np.maximum(0, df['outdoor_temp'] - 24)
        df['temp_squared'] = df['outdoor_temp'] ** 2
    
    if 'solar_irradiance' in df.columns:
        df['solar_log'] = np.log1p(df['solar_irradiance'])
    
    # Interaction features
    if 'floor_area' in df.columns and 'outdoor_temp' in df.columns:
        df['area_temp_interaction'] = df['floor_area'] * df['outdoor_temp']
    
    if 'window_ratio' in df.columns and 'solar_irradiance' in df.columns:
        df['window_solar_interaction'] = df['window_ratio'] * df['solar_irradiance']
    
    # Time-based features (if hour column exists)
    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_year'] = (df['hour'] // 24) % 365
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return df

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive model evaluation metrics."""
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'max_error': np.max(np.abs(y_true - y_pred)),
        'std_error': np.std(y_true - y_pred)
    }
    
    return metrics

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def detect_outliers(data: pd.DataFrame, columns: List[str] = None,
                   method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """Detect outliers in the dataset."""
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    outlier_mask = pd.Series(False, index=data.index)
    
    for col in columns:
        if method == 'iqr':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            col_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            col_outliers = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        outlier_mask |= col_outliers
    
    return data[~outlier_mask]

def create_feature_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics for features."""
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    summary = pd.DataFrame({
        'count': data[numeric_cols].count(),
        'mean': data[numeric_cols].mean(),
        'std': data[numeric_cols].std(),
        'min': data[numeric_cols].min(),
        '25%': data[numeric_cols].quantile(0.25),
        '50%': data[numeric_cols].quantile(0.50),
        '75%': data[numeric_cols].quantile(0.75),
        'max': data[numeric_cols].max(),
        'missing': data[numeric_cols].isnull().sum(),
        'missing_pct': (data[numeric_cols].isnull().sum() / len(data)) * 100
    })
    
    return summary.round(3)

def split_time_series(data: pd.DataFrame, 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time series data maintaining temporal order."""
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

def calculate_energy_metrics(predictions: Dict[str, np.ndarray],
                           actuals: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Calculate energy-specific metrics."""
    
    metrics = {}
    
    for energy_type in predictions.keys():
        y_true = actuals[energy_type]
        y_pred = predictions[energy_type]
        
        # Standard metrics
        base_metrics = evaluate_model(y_true, y_pred)
        
        # Energy-specific metrics
        total_actual = np.sum(y_true)
        total_predicted = np.sum(y_pred)
        
        energy_metrics = {
            **base_metrics,
            'total_actual': total_actual,
            'total_predicted': total_predicted,
            'total_error_pct': ((total_predicted - total_actual) / total_actual) * 100,
            'peak_actual': np.max(y_true),
            'peak_predicted': np.max(y_pred),
            'peak_error_pct': ((np.max(y_pred) - np.max(y_true)) / np.max(y_true)) * 100
        }
        
        metrics[energy_type] = energy_metrics
    
    return metrics

def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and return issues."""
    
    issues = {
        'missing_data': {},
        'negative_values': {},
        'outliers': {},
        'data_types': {},
        'summary': {}
    }
    
    # Check missing data
    missing = data.isnull().sum()
    issues['missing_data'] = missing[missing > 0].to_dict()
    
    # Check for negative values in energy columns
    energy_cols = ['heating_load', 'cooling_load', 'total_energy']
    for col in energy_cols:
        if col in data.columns:
            negative_count = (data[col] < 0).sum()
            if negative_count > 0:
                issues['negative_values'][col] = negative_count
    
    # Check data types
    for col in data.columns:
        if data[col].dtype == 'object':
            issues['data_types'][col] = 'categorical'
    
    # Summary
    issues['summary'] = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(data.select_dtypes(include=['object']).columns),
        'missing_data_pct': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
    }
    
    return issues