import random
import torch
import numpy as np
import torch.nn as nn
from gensim.models import Word2Vec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WordLSTM(nn.Module):

    def __init__(self, hidden_size, n_layers):
        super(WordLSTM, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.softmax = nn.Softmax(dim=1)

        training_text = list(map(lambda s: s.split(" "), open("./data/bach.txt").read().splitlines()))
        self.encoder = Word2Vec(training_text, vector_size=hidden_size, seed=7, window=5, min_count=1, workers=4)
        self.decoder = nn.Linear(hidden_size, len(self.encoder.wv))

    def forward(self, input, hidden):  # Where hidden is (hidden_state, memory_cell)
        output, hidden = self.lstm(input, hidden)
        return self.decoder(output), hidden

    def init_hidden(self):
        return (torch.zeros(self.n_layers, self.hidden_size),  # Hidden State
                torch.zeros(self.n_layers, self.hidden_size))  # Memory Cell


def train(input, target, criterion, optimizer, net):
    total_loss = 0
    hidden = net.init_hidden()
    embeddings = net.encoder.wv

    for si in range(len(input)):
        word_input_list = torch.as_tensor(np.array(list(
            map(lambda w: embeddings[w],
                input[si].decode("utf-8").split(" ")))
        ))
        word_target = torch.as_tensor(embeddings.key_to_index[target[si].decode("utf-8")]).unsqueeze(0)

        optimizer.zero_grad()

        for word_input in word_input_list[:-1]:
            _, hidden = net(word_input.unsqueeze(0), hidden)

        output, hidden = net(word_input_list[-1].unsqueeze(0), hidden)
        loss = criterion(output, word_target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Detach hidden state to prevent backpropagation through time
        hidden = tuple([h.detach() for h in hidden])  # ChatGPT

    return total_loss/len(input)


def training_set(sentences):
    input = np.chararray(shape=len(sentences), itemsize=500)
    target = np.chararray(shape=len(sentences), itemsize=500)

    for si in range(len(sentences)):
        sentence = sentences[random.randint(0, len(sentences) - 1)]
        sentence_list = sentence.split(" ")

        input[si] = sentence[:-(len(sentence_list[-1]) + 1)]
        target[si] = sentence_list[-1]

    return input, target


def generate(net, prime_str="The child was", predict_len=100):
    hidden = net.init_hidden()
    prime_input = prime_str.split(" ")

    embeddings = net.encoder.wv

    # Use priming string to "build up" hidden state
    for word in prime_input[:-1]:
        _, hidden = net(torch.as_tensor(np.copy(embeddings[word])).unsqueeze(0), hidden)

    word_input = prime_input[-1]
    word_output = prime_str

    for i in range(predict_len):

        word_embedding = torch.as_tensor(np.copy(embeddings[word_input])).unsqueeze(0)
        output, hidden = net(word_embedding, hidden)

        word_input = embeddings.index_to_key[np.argmax(output.detach())]
        word_output += " " + word_input

        if len(word_output) > 500:
            word_output += "\n"

    return word_output
