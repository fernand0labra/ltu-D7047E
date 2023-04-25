import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import utils



# Define execution device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n###########################################################################")
print("#                   CNN Small Training on MNIST dataset                   #")
print("###########################################################################\n")

# Hyperparameters
batch_size = 512
epochs = 1
learning_rate = 0.000001

# Define transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the training and test datasets
trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

net = utils.CNN()
net.cuda(device)  # GPU

log_directory = './logs/cnn_mnist_small'
utils.train(epochs, trainloader, optim.Adam(net.parameters(), lr=learning_rate), nn.CrossEntropyLoss(), net, device, log_directory)

PATH = 'models/cnn_mnist_small.pth'
torch.save(net.state_dict(), PATH)


print("\n###########################################################################")
print("#                  CNN Medium Training on MNIST dataset                   #")
print("###########################################################################\n")

# Hyperparameters
epochs = 15
learning_rate = 0.001

log_directory = './logs/cnn_mnist_medium'
utils.train(epochs, trainloader, optim.Adam(net.parameters(), lr=learning_rate), nn.CrossEntropyLoss(), net, device, log_directory)

PATH = 'models/cnn_mnist_medium.pth'
torch.save(net.state_dict(), PATH)