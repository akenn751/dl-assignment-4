from src.models.rnn_vanilla import VanillaRNN
import torch

model = VanillaRNN(input_size=100, hidden_size=64, output_size=100)
inputs = torch.randint(0, 100, (10, 1))  # (T=10, B=1)

logits, h = model(inputs)
print(logits.shape)  # should be (10, 1, 100)
