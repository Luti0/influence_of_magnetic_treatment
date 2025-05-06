import torch
import numpy as np
import os
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from scripts.model_nn import AdsorptionNet

def train_model(
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    train_loader: Optional[DataLoader] = None,
    epochs: int = 300,
    lr: float = 1e-4,
    batch_size: int = 16,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> nn.Module:
    """
    Training a neural network on data from DataLoader.
    """

    if train_loader is None:
        if X_train is None or y_train is None:
            raise ValueError("Either train_loader or (X_train and y_train) must be provided.")

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AdsorptionNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        if verbose:
            print(f"Model saved: {save_path}")

    return model
