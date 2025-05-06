import torch
import numpy as np

class TorchModelWrapper:
    def __init__(self, model):
        self.model = model.eval()

    def predict(self, X):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, np.ndarray) and X.ndim == 1:
            X = X.reshape(1, -1)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_tensor = self.model(X_tensor)
        return y_tensor.numpy()