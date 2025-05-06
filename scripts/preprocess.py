import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
from typing import Tuple, Optional
import numpy as np
import sys

def file_to_pd(file_path: str) -> pd.DataFrame:
    """
    File path to DataFrame pandas.
    Columns renaming in DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        new_columns = ['Bm', 'k', 'l', 'r', 'J']
        if len(df.columns) == len(new_columns):
            df.columns = new_columns
        else:
            print(f"Warning: The number of columns in the file ({len(df.columns)}) does not match the expected number ({len(new_columns)}). Columns were not renamed.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at path {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)



def mlp_preprocess(
    df: pd.DataFrame,
    use_scaler: bool = True,
    test_size: float = 0.3,
    batch_size: int = 8
    ) -> Tuple[DataLoader, DataLoader, Optional[StandardScaler], Optional[StandardScaler]]:
    
    """
    Preprocess for MLP (PyTorch).
    Standardization of features and creation of DataLoaders.
    """

    X = df[['Bm', 'k', 'l', 'r']].values
    y = df['J'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

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


def preprocess_for_ML(
    df: pd.DataFrame, 
    test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Preprocessing data for Random Forest or other tree models.
    Without feature standardization.
    """

    x = df[['Bm', 'k', 'l', 'r']].values
    y = df['J'].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    return X_train, X_test, y_train, y_test