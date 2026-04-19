from src.tests.rnn_tiny import TinyVanillaRNN
import torch
import torch.nn as nn

model = TinyVanillaRNN(5, 4, 5)
x = torch.tensor([0,1,2])
targets = torch.tensor([1,2,3])

loss = nn.CrossEntropyLoss()(model(x)[0], targets)
loss.backward()

for name, p in model.named_parameters():
    print(name, p.grad)
