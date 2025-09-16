"""
utils.py
--------
Helper functions for saving/loading models, scalers, etc.
"""

import joblib


def save_model(obj, filename: str):
    joblib.dump(obj, filename)


def load_model(filename: str):
    return joblib.load(filename)
