import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

def data_preprocess(
    df: pd.DataFrame,
    model_name: str = 'mlp',
    test_size: float = 0.2,
    use_scaler: bool = True,
    batch_size: int = 8,
    random_state: int = 42
    ):
    """
    Preprocess for models.
    Split for ML models.
    Standardization of features and creation of DataLoaders for MLP.
    """
    X = df[['Bm', 'k', 'l', 'r']].values
    y = df['J'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_name != 'mlp':
        return X_train, X_test, y_train, y_test


    if use_scaler:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)
    else:
        scaler_X = None
        scaler_y = None

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler_X, scaler_y
