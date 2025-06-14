import torch.nn as nn
import torch
import numpy as np
'''
MLP (3 hidden layer 64 neurons in each).
'''
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(4, 64)  
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.output(x)
        return x

    def predict(self, X):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, np.ndarray) and X.ndim == 1:
            X = X.reshape(1, -1)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            y_tensor = self(X_tensor)
        return y_tensor.numpy()
    

def load_model(path: str):
    model = MLP()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model