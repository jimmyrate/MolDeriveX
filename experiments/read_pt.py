import torch
from botorch.models import MultiTaskGP

train_X = torch.rand(20, 2)  # 20 points in 2 dimensions
train_Y = torch.stack([torch.sin(train_X[:, 0] * (2 * torch.pi)), torch.cos(train_X[:, 1] * (2 * torch.pi))], -1)  # 2 tasks

gp = MultiTaskGP(train_X=train_X, train_Y=train_Y,)