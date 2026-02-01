"""
HVAC System Project - Engineering Calculations

Core HVAC engineering calculations. Some formulas adapted from
open source references and ASHRAE handbooks.

v0.2.0 - added ML module, dashboard (Bola)
v0.1.x - core engineering calcs
"""
__version__ = "0.2.0"
__author__ = "Bola"

from .pint_setup import Quantity, UNITS
from .misc import print_doc_string

# ML module needs sklearn/xgboost - make it optional
try:
    from . import ml
except ImportError:
    ml = None  # that's fine, not everyone needs ML
