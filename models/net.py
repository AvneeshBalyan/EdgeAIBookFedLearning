# models/net.py
import torch.nn as nn

class ArrhythmiaNet(nn.Module):
    def __init__(self):
        super(ArrhythmiaNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(2, 64),  # 2 features: MLII and V5
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Linear(32, 6)  # Changed to 6 classes (0-5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
