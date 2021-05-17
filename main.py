from stl_10_patchpair import DatasetForPretext
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import ContextPredictionModel
import numpy as np

"""
    dataset[idx] : center, other, label 이런 형식
"""
train_dataset = DatasetForPretext()
train_dataloader = DataLoader(dataset=train_dataset,batch_size=32, shuffle=True)

"""
    Model
"""

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
model = ContextPredictionModel().to(DEVICE)
optim = torch.optim.Adam(params=model.parameters(),lr = 1e-3)
criterion = nn.CrossEntropyLoss()

"""
    Train sample
"""

for epoch in range(30):
    losses = []
    print("{}/{} Epoch".format(epoch + 1, 30))
    for center, other, label in tqdm(train_dataloader):
        center = center.to(DEVICE)
        other = other.to(DEVICE)
        label = label.to(DEVICE)
        pred = model(center, other)
        loss = criterion(pred, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())
    print("Loss : {}".format(np.mean(losses)))