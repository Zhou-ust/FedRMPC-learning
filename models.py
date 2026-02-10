# models.py
import torch
import torch.nn as nn
from config import Config


class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, Config.HIDDEN_DIM)
        self.fc2 = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        self.fc3 = nn.Linear(Config.HIDDEN_DIM, 4)
        self.dropout = nn.Dropout(p=Config.DROPOUT)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

    def predict_uncertainty(self, s_np, a_np, num_samples=10):
        self.train()
        s = torch.FloatTensor(s_np).unsqueeze(0).to(Config.DEVICE)
        a = torch.FloatTensor(a_np).unsqueeze(0).to(Config.DEVICE)

        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(self.forward(s, a))

        preds = torch.stack(preds)
        mean = preds.mean(dim=0).squeeze(0).cpu().numpy()
        var = preds.var(dim=0).sum().item()
        return mean, var