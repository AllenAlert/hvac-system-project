"""
ML module for HVAC prediction.

Train models on simulation or historical data for quick energy/load estimates.

Requires: scikit-learn, xgboost (optional), joblib

@author: Bola
"""

from .data_generator import HVACDataGenerator
from .models import EnergyPredictor, LoadPredictor
from .trainer import ModelTrainer
from .utils import prepare_features, evaluate_model

__all__ = [
    'HVACDataGenerator',
    'EnergyPredictor', 
    'LoadPredictor',
    'ModelTrainer',
    'prepare_features',
    'evaluate_model'
]