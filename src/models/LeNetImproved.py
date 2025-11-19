# import torch
import torch.nn as nn
# import torch.nn.functional as F

class LeNetImproved(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    # kernel 2x2; 28x28 -> 14x14
            
            nn.Conv2d(32,64,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 14x14 -> 7x7
        
            nn.Conv2d(64,128,3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 120), # 7x7 as (28x28)/2/2
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
    )
    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x    # logits