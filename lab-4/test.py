import os
import torch
import utils

model = "char_shakespeare_gru.pt"
predict_len = "100"
temperature = "0.7"

random_primes = ["2 b3n", "bg09Z", "3,0si"]

for prime_str in random_primes:
    print("\n\n#####################################################")
    print("Prime String: {}\n".format(prime_str))
    # Generate char sequences from the model
    os.system("python ./char_rnn_pytorch/generate.py ./models/" + model +
              " --prime_str=\"" + prime_str +
              "\" --predict_len=" + predict_len +
              " --temperature=" + temperature +
              " --cuda")

coherent_primes = ["The", "What is", "Shall I give", "X087hNYB BHN BYFVuhsdbs"]

for prime_str in coherent_primes:
    print("\n\n#####################################################")
    print("Prime String: {}\n".format(prime_str))
    # Generate char sequences from the model
    os.system("python ./char_rnn_pytorch/generate.py ./models/" + model +
              " --prime_str=\"" + prime_str +
              "\" --predict_len=" + predict_len +
              " --temperature=" + temperature +
              " --cuda")

net = torch.load("./models/word_bach_lstm.pt")

word_primes = ["The", "which is", "He he he", "He was the", "Leipzig is known as"]
for prime_str in word_primes:
    print(utils.generate(net, prime_str, predict_len=3))

os.system("tensorboard --logdir=\"./logs\"")