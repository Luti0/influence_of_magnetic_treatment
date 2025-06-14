from torch import nn, optim
from torch.utils.data import DataLoader
import torch

def train_mlp(
    model,
    train_loader: DataLoader = None,
    epochs: int = 2000,
    PATH: str = '../models/mlp_weights.pth'
):
    '''
    Training for MLP model.
    '''
    
    if train_loader is None:
        raise ValueError('No training parameters')
    
    model = model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-3)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), PATH)
    return model


from joblib import dump
import optuna
from sklearn.model_selection import cross_val_score

def train_ML(
    model_cls,
    X_train = None,
    y_train = None,
    random_state: int = 42,
    n_trials: int = 200,
    cv_folds: int = 5,
    scoring: str = 'neg_mean_squared_error',
    PATH: str = '../models/ML_weights.pth'
):
    '''
    Training for ML models and optimize with optuna.
    '''
    if X_train is None or y_train is None:
        raise ValueError('No training parameters')
    
    def optuna_optimize(trial):
        if model_cls.__name__ == 'RandomForestRegressor':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state
            }
        elif model_cls.__name__ == 'XGBRegressor':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.2, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
                'random_state': random_state
            }
        else:
            raise NotImplementedError(f'Optuna search space not implemented for {model.__name__}')
        
        model = model_cls(**params)
        scores = cross_val_score(model, X_train, y_train.ravel(), cv=cv_folds, scoring=scoring)
        return scores.mean()


    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_optimize, n_trials=n_trials)

    best_params = study.best_params
    best_params['random_state'] = random_state

    final_model = model_cls(**best_params)
    final_model.fit(X_train,y_train)

    dump(final_model, PATH)
    return final_model