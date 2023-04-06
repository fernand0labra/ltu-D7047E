import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import utils


# Hyperparameters
batch_size = 128
epochs = 100
learning_rate = 0.0001

# Load and normalize CIFAR10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

# Define loss criterion
criterion = nn.CrossEntropyLoss()

# Define execution device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nTRAINING PHASE\n")

print("\n###########################################################################")
print("# CNN with LeakyReLu activation and Stochastic Gradient Descent optimizer #")
print("###########################################################################\n")

net = utils.Net(activation="LeakyReLu")
net.cuda(device)  # GPU

log_directory = './logs/cifar_sgd_leaky_relu'
utils.train(epochs, trainloader, optim.SGD(net.parameters(), lr=learning_rate), criterion, net, device, log_directory)

PATH = './models/cifar_sgd_leaky_relu.pth'
torch.save(net.state_dict(), PATH)

print("\n###########################################################################")
print("#           CNN with LeakyReLu activation and Adam optimizer              #")
print("###########################################################################\n")

net = utils.Net(activation="LeakyReLu")
net.cuda(device)  # GPU

log_directory = './logs/cifar_adam_leaky_relu'
utils.train(epochs, trainloader, optim.Adam(net.parameters(), lr=learning_rate), criterion, net, device, log_directory)

PATH = './models/cifar_adam_leaky_relu.pth'
torch.save(net.state_dict(), PATH)

print("\n###########################################################################")
print("#              CNN with Tanh activation and Adam optimizer                #")
print("###########################################################################\n")

net = utils.Net(activation="Tanh")
net.cuda(device)  # GPU

log_directory = './logs/cifar_adam_tanh'
utils.train(epochs, trainloader, optim.Adam(net.parameters(), lr=learning_rate), criterion, net, device, log_directory)

PATH = './models/cifar_adam_tanh.pth'
torch.save(net.state_dict(), PATH)
