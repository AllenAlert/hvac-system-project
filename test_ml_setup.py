"""
Quick test script for HVAC ML framework.
Run this to verify the setup is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ml_setup():
    print("Testing HVAC ML Framework Setup...")
    
    try:
        # Test imports
        from hvac.ml import HVACDataGenerator, ModelTrainer, EnergyPredictor
        print("✓ ML modules imported successfully")
        
        # Test data generation
        generator = HVACDataGenerator()
        data = generator.generate_building_dataset(n_samples=100)
        print(f"✓ Generated {len(data)} samples")
        
        # Test model creation
        trainer = ModelTrainer()
        model = trainer.train_energy_model(data, model_type='linear')
        print("✓ Model trained successfully")
        
        # Test prediction
        test_sample = data.iloc[:1]
        feature_cols = [col for col in data.columns if col not in 
                       ['heating_load', 'cooling_load', 'total_energy']]
        prediction = model.predict(test_sample[feature_cols])
        print(f"✓ Prediction made: {prediction[0]:.2f} kW")
        
        print("\n🎉 All tests passed! ML framework is ready to use.")
        print("\nNext steps:")
        print("1. Run: python examples/ml_demo.py")
        print("2. Install ML dependencies: pip install -r requirements-ml.txt")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements-ml.txt")
        print("2. Check Python path and imports")

if __name__ == "__main__":
    test_ml_setup()