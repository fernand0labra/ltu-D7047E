import math
import os
import string

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = "lstm"
n_epochs = "2000"
print_every = "200"
hidden_size = "50"
n_layers = "2"
learning_rate = "0.01"
chunk_len = "200"
batch_size = "1"

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


net = utils.WordLSTM(int(hidden_size), int(n_layers))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

writer = SummaryWriter("./logs")
sentences = open("./data/bach.txt").read().splitlines()

for epoch in tqdm(range(1, int(n_epochs) + 1)):
    input, target = utils.training_set(sentences)

    loss = utils.train(input, target, criterion, optimizer, net)
    writer.add_scalar("Loss/train", math.exp(loss), epoch)  # Add perplexity

    if epoch % int(print_every) == 0:
        print('[(%d %d%%) %.4f]' % (epoch, epoch / int(n_epochs) * 100, loss))

torch.save(net, "./models/word_bach_lstm.pt")
