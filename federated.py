# federated.py
import torch
import torch.optim as optim
import numpy as np
from config import Config
from models import BNN


class FedClient:
    def __init__(self, id):
        self.id = id
        self.model = BNN().to(Config.DEVICE)
        self.data_buffer = []
        self.initial_weights = None  # 用于计算漂移

    def add_data(self, s, a, s_next):
        self.data_buffer.append((s, a, s_next - s))

    def train(self, global_weights, mode='FedProx'):
        """返回: (test_mse, updated_weights, drift)"""
        self.model.load_state_dict(global_weights)
        self.initial_weights = copy_weights(self.model)

        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=Config.LR)
        loss_fn = torch.nn.MSELoss()

        if len(self.data_buffer) < 5:
            return 0.0, self.model.state_dict(), 0.0

        # 数据准备
        s = torch.FloatTensor(np.array([i[0] for i in self.data_buffer])).to(Config.DEVICE)
        a = torch.FloatTensor(np.array([i[1] for i in self.data_buffer])).to(Config.DEVICE)
        d = torch.FloatTensor(np.array([i[2] for i in self.data_buffer])).to(Config.DEVICE)

        # 1. 计算 Test MSE
        with torch.no_grad():
            pred_test = self.model(s, a)
            test_mse = loss_fn(pred_test, d).item()

        # 2. 本地训练
        for _ in range(Config.LOCAL_EPOCHS):
            opt.zero_grad()
            pred = self.model(s, a)
            mse = loss_fn(pred, d)
            loss = mse
            if mode == 'FedProx':
                prox = 0
                for w, w_g in zip(self.model.parameters(), global_weights.values()):
                    prox += (w - w_g).norm(2) ** 2
                loss += (Config.PROXIMAL_MU / 2) * prox
            loss.backward()
            opt.step()

        # 3. 计算 Client Drift
        drift = calc_diff(self.model, self.initial_weights)

        return test_mse, self.model.state_dict(), drift


def copy_weights(model):
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def calc_diff(model, weights_dict):
    diff = 0
    for k, v in model.state_dict().items():
        diff += (v - weights_dict[k]).norm(2).item()
    return diff


class FedServer:
    def __init__(self):
        self.global_model = BNN().to(Config.DEVICE)

    def aggregate(self, clients):
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            weights = torch.stack([c.model.state_dict()[k] for c in clients])
            global_dict[k] = weights.mean(0)
        self.global_model.load_state_dict(global_dict)
        return global_dict

    def calc_model_variance(self, clients):
        w_g = self.global_model.state_dict()
        diff = 0
        for c in clients:
            w_l = c.model.state_dict()
            for k in w_g.keys():
                diff += (w_l[k] - w_g[k]).norm(2).item()
        return diff / len(clients)