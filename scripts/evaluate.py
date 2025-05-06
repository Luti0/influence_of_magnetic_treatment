import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Union, Callable
from sklearn.preprocessing import StandardScaler
from scripts.torch_wrapper import TorchModelWrapper
import sklearn, torch
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV

def evaluate_model(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    threshold: float = 0.00005
    ) -> Dict[str, float]:
    
    '''
    Evaluates a model by computing regression metrics.
    '''
    
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



def k_fold_cross_val(
    df: pd.DataFrame,
    model_fn: Callable[[], Union['sklearn.base.BaseEstimator', 'torch.nn.Module']],
    use_scaling: bool = True,
    k_folds: int = 5,
    threshold: float = 0.0001,
    use_torch_wrapper: bool = False,
    verbose: bool = True,
    return_fold_metrics: bool = False,
    use_gridsearch: bool = True
) -> Dict[str, float]:
    """
    K-Fold cross val.
    """

    X = df[['Bm', 'k', 'l', 'r']].values
    y = df['J'].values.reshape(-1, 1)

    kf = KFold(n_splits=k_folds, shuffle=True)
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if use_scaling:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            X_val_scaled   = scaler_X.transform(X_val)

            y_train_scaled = scaler_y.fit_transform(y_train)
            y_val_scaled   = scaler_y.transform(y_val)
        else:
            scaler_X = None
            scaler_y = None
            X_train_scaled, X_val_scaled = X_train, X_val
            y_train_scaled, y_val_scaled = y_train, y_val

        model = model_fn()

        if hasattr(model, 'fit') and use_gridsearch:
            model_class = model.__class__.__name__

            if model_class == "RandomForestRegressor":
                param_grid = {
                    'n_estimators': range(100, 500),
                    'max_depth': range(2, 10),
                    'min_samples_leaf': range(2, 8)
                }
            elif model_class == "XGBRegressor":
                param_grid = {
                    'n_estimators': range(100, 500),
                    'max_depth': range(2, 10),
                    'learning_rate': [0.01, 0.005, 0.001, 0.0005]
                }
            else:
                param_grid = None

            if param_grid:
                search = RandomizedSearchCV(model, param_grid, cv=4, scoring='neg_mean_squared_error', n_iter=250)
                search.fit(X_train_scaled, y_train_scaled.ravel())
                model = search.best_estimator_

        elif hasattr(model, 'fit'):
            model.fit(X_train_scaled, y_train_scaled.ravel())
        else:
            from scripts.train_nn import train_model
            model = train_model(X_train=X_train_scaled, y_train=y_train_scaled, epochs=300, verbose=False)

        if use_torch_wrapper:
            model = TorchModelWrapper(model)

        X_val_input = X_val_scaled
        y_pred_scaled = model.predict(X_val_input)

        if scaler_y:
            y_val_orig = scaler_y.inverse_transform(y_val_scaled)
            y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)
        else:
            y_val_orig = y_val_scaled
            y_pred_orig = y_pred_scaled

        metrics = evaluate_model(y_val_orig, y_pred_orig, threshold)
        all_metrics.append(metrics)

        if verbose:
            print(f"[Fold {fold+1}/{k_folds}] MSE={metrics['MSE']:.6f}, Accuracy={metrics[f'Accuracy < {threshold}']:.2f}%")

    df_metrics = pd.DataFrame(all_metrics)

    if return_fold_metrics:
        return df_metrics  

    return df_metrics.mean().to_dict() 