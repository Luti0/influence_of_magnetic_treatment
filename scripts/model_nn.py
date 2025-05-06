import torch.nn as nn

'''
MLP (3 hidden layer 64 neurons in each).
'''

class AdsorptionNet(nn.Module):
    def __init__(self):
        super(AdsorptionNet, self).__init__()
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
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.output(x)
        return x