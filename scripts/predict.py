import torch
import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Union
import sklearn
from sklearn.model_selection import KFold

# Path to model
MODEL_PATH = '../models/adorption_model.pth'

def predict_single(
    model: Union[torch.nn.Module, 'sklearn.base.BaseEstimator'],
    params: Union[List[float], np.ndarray],
    scaler_X: Optional[StandardScaler] = None,
    scaler_y: Optional[StandardScaler] = None
) -> float:
    '''
    Model prediction for one parameter set.
    '''
    params = [int(params[0])] + list(params[1:])
    params = np.array(params).reshape(1, -1)

    if scaler_X is not None:
        params = scaler_X.transform(params)

    prediction = model.predict(params)

    if scaler_y is not None:
        prediction = scaler_y.inverse_transform(prediction)

    return float(prediction[0])


def find_min_J(
    model: Union[torch.nn.Module, 'sklearn.base.BaseEstimator'],
    bounds: List[Tuple[float, float]],
    scaler_X: Optional[StandardScaler] = None,
    scaler_y: Optional[StandardScaler] = None,
    steps_per_float: int = 100
) -> Tuple[float, List[float]]:
    '''
    GreedSearch for parameters minimizing the prediction of the model.
    '''
    int_bound = bounds[0]  
    float_bounds = bounds[1:]  

    int_range = np.arange(int_bound[0], int_bound[1] + 1, 1, dtype=int)

    float_ranges = [
        np.linspace(b[0], b[1], steps_per_float) for b in float_bounds
    ]

    min_value = float('inf')
    best_params = None

    for int_param in int_range:
        for float_params in itertools.product(*float_ranges):

            params = [int_param] + list(float_params)
            value = predict_single(model, params, scaler_X, scaler_y)

            if value < min_value:
                min_value = value
                best_params = params

    return best_params, min_value
