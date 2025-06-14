from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.models.mlp import MLP

def get_model(name):
    name = name.lower()
    if name == "rf" or name == 'random forest' or name == 'forest':
        return RandomForestRegressor
    elif name == "xgb" or name == 'xgboost':
        return XGBRegressor
    elif name == "mlp" or name == 'nn':
        return MLP()
    else:
        raise ValueError(f"Unknown model: {name}")