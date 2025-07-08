import pandas as pd
import torch
import torch.nn as nn

class FullyConnectedRegression(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        
    def forward(self, x):
        return self.linear_relu_stack(x)

def loss_fn(y_pred, y_true):
    mae = torch.mean(torch.abs(y_pred - y_true))
    abs_diff = torch.abs(y_pred - y_true)
    return abs_diff

def train_test_split(x_tensor, y_tensor, test_size=0.2):
    indices = torch.randperm(len(x_tensor))
    split_idx = int(len(x_tensor) * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return x_tensor[train_indices], y_tensor[train_indices], x_tensor[test_indices], y_tensor[test_indices]

data = pd.read_csv("train.csv")
# x = torch.tensor(data['SMILES'])
y = torch.tensor(data[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy())
# print(x[:10])
print(y[:10])







