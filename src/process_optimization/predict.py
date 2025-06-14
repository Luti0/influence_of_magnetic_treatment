import numpy as np
import itertools
from typing import List, Tuple
from scipy.optimize import differential_evolution

def predict_single(
    model,
    params,
    scaler_X = None,
    scaler_y = None
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
    model,
    bounds,
    scaler_X = None,
    scaler_y = None,
) -> Tuple[float, List[float]]:
    '''
    GreedSearch for parameters minimizing the prediction of the model.
    '''

    def objective(params):
        int_param = int(round(params[0]))
        full_params = [int_param] + list(params[1:])
        return predict_single(model, full_params, scaler_X, scaler_y)
    
    float_bounds = [(b[0], b[1]) for b in bounds]
    
    result = differential_evolution(
        objective,
        bounds=float_bounds,
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=1e-6,
        seed=42
    )

    int_param = int(round(result.x[0]))
    final_params = [int_param] + list(result.x[1:])
    return final_params, result.fun
