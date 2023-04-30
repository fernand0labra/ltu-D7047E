import random
import torch
import torch.nn as nn
from char_rnn_pytorch.helpers import char_tensor, all_characters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(CharLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.n_layers = n_layers

        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):  # Where hidden is (hidden_state, memory_cell)
        input = self.encoder(input)
        output, hidden = self.lstm(input.unsqueeze(1), hidden)
        return self.decoder(output.view(input.size(0), -1)), hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device),  # Hidden State
                torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device))  # Memory Cell


def train(input, target, criterion, optimizer, batch_size, net):
    hidden = net.init_hidden(batch_size)
    loss = 0
    optimizer.zero_grad()

    for c in range(len(input[0])):
        output, hidden = net(input[:, c], hidden)
        loss += criterion(output, target[:, c])
    # loss.backward()
    optimizer.step()

    return loss.data/len(input)


def random_training_set(file, file_len, chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)

    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1

        chunk = file[start_index:end_index]

        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])

    return inp, target


def generate(net, prime_str='A', predict_len=100, temperature=0.8):
    hidden = torch.zeros(net.hidden_size)
    prime_input = char_tensor(prime_str).unsqueeze(0)

    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = net(prime_input[:, p], hidden)

    inp = prime_input[:, -1]

    for p in range(predict_len):
        output, hidden = net(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char).unsqueeze(0)

    return predicted
