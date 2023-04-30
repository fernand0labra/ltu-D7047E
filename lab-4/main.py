import os

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = "lstm"
n_epochs = "2000"
print_every = "500"
hidden_size = "50"
n_layers = "2"
learning_rate = "0.01"
chunk_len = "200"
batch_size = "128"

# Train the charRNN model on the shakespeare data
os.system("python ./char_rnn_pytorch/train.py ./data/shakespeare.txt"
          " --model=" + model +
          " --n_epochs=" + n_epochs +
          " --print_every=" + print_every +
          " --hidden_size=" + hidden_size +
          " --n_layers=" + n_layers +
          " --learning_rate=" + learning_rate +
          " --chunk_len=" + chunk_len +
          " --batch_size=" + batch_size +
          " --cuda")