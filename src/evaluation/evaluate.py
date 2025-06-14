import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict


def evaluate_model(
    y_true = None,
    y_pred = None,
    threshold: float = 0.005
    ) -> Dict[str, float]:
    
    '''
    Evaluates a model by computing regression metrics.
    '''

    if y_true is None or y_pred is None:
        raise ValueError('No evaluating parameters') 
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    threshold_acc = np.mean(np.abs(y_true - y_pred) < threshold) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'R²': r2,
        f'Accuracy < {threshold}': threshold_acc
    }