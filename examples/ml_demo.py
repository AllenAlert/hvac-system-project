"""
Example script demonstrating HVAC ML framework usage.
This script shows how to generate data, train models, and make predictions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hvac.ml import HVACDataGenerator, ModelTrainer, EnergyPredictor, LoadPredictor
from hvac.ml.utils import prepare_features, validate_data_quality

def main():
    print("HVAC Machine Learning Framework Demo")
    print("=" * 50)
    
    # 1. Generate training data
    print("\n1. Generating training data...")
    generator = HVACDataGenerator()
    
    # Generate building dataset
    building_data = generator.generate_building_dataset(n_samples=2000)
    print(f"Generated {len(building_data)} building samples")
    print(f"Features: {list(building_data.columns)}")
    
    # Validate data quality
    quality_report = validate_data_quality(building_data)
    print(f"Data quality: {quality_report['summary']}")
    
    # 2. Train energy prediction model
    print("\n2. Training energy prediction models...")
    trainer = ModelTrainer()
    
    # Compare different models
    comparison = trainer.compare_models(
        building_data, 
        target_col='total_energy',
        models=['linear', 'random_forest', 'xgboost']
    )
    
    # Get best model
    best_model_type = comparison.iloc[0]['model_type']
    best_model = trainer.results[f'energy_{best_model_type}']['model']
    
    print(f"\nBest model: {best_model_type}")
    print(f"Feature importance:")
    importance = trainer.results[f'energy_{best_model_type}']['feature_importance']
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feature}: {imp:.4f}")
    
    # 3. Train load prediction model
    print("\n3. Training load prediction model...")
    load_model = trainer.train_load_model(building_data, model_type='random_forest')
    
    # 4. Generate time series data for a specific building
    print("\n4. Generating time series data...")
    building_params = {
        'floor_area': 1000,
        'height': 3.0,
        'window_ratio': 0.3,
        'insulation_r': 3.5,
        'occupancy': 20
    }
    
    time_series_data = generator.generate_time_series(building_params, hours=8760)
    print(f"Generated {len(time_series_data)} hourly data points")
    
    # 5. Train time series model
    print("\n5. Training time series model...")
    ts_model = trainer.train_time_series_model(time_series_data, lookback_hours=24)
    
    # 6. Make predictions on new data
    print("\n6. Making predictions...")
    
    # Create test building
    test_building = pd.DataFrame({
        'floor_area': [1500],
        'height': [3.2],
        'window_ratio': [0.4],
        'insulation_r': [4.0],
        'occupancy': [25],
        'outdoor_temp': [15],
        'solar_irradiance': [500],
        'wind_speed': [5]
    })
    
    # Energy prediction
    energy_pred = best_model.predict(test_building)[0]
    print(f"Predicted total energy: {energy_pred:.2f} kW")
    
    # Load predictions
    heating_pred, cooling_pred = load_model.predict(test_building)
    print(f"Predicted heating load: {heating_pred[0]:.2f} kW")
    print(f"Predicted cooling load: {cooling_pred[0]:.2f} kW")
    
    # 7. Save models
    print("\n7. Saving models...")
    best_model.save('energy_model.joblib')
    load_model.save('load_model.joblib')
    print("Models saved successfully!")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()