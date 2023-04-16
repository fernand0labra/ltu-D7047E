import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import utils

# Hyperparameters
batch_size = 512
epochs = 10
learning_rate = 0.0001

# Load and normalize CIFAR10
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

# Define loss criterion
criterion = nn.CrossEntropyLoss()

# Define execution device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nTRAINING PHASE\n")


print("\n###########################################################################")
print("#                  AlexNet Fine Tuning on CIFAR-10 dataset                #")
print("###########################################################################\n")

net = utils.AlexNet("fine_tuning")
net.cuda(device)  # GPU

log_directory = './logs/alexnet_fine_tuning_cifar10'
utils.train(epochs, trainloader, optim.Adam(net.parameters(), lr=learning_rate), criterion, net, device, log_directory)

PATH = 'models/alexnet_fine_tuning_cifar10.pth'
torch.save(net.state_dict(), PATH)


print("\n###########################################################################")
print("#            AlexNet Feature Extraction on CIFAR-10 dataset               #")
print("###########################################################################\n")

learning_rate = 0.01

net = utils.AlexNet("feature_extraction")
net.cuda(device)  # GPU

log_directory = './logs/alexnet_feature_extraction_cifar10'
utils.train(epochs, trainloader, optim.Adam(net.parameters(), lr=learning_rate), criterion, net, device, log_directory)

PATH = 'models/alexnet_feature_extraction_cifar10.pth'
torch.save(net.state_dict(), PATH)


print("\n###########################################################################")
print("#                      Own-CNN Training on MNIST dataset                  #")
print("###########################################################################\n")

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

log_directory = './logs/cnn_mnist'
utils.train(epochs, trainloader, optim.Adam(net.parameters(), lr=learning_rate), criterion, net, device, log_directory)

PATH = 'models/cnn_mnist.pth'
torch.save(net.state_dict(), PATH)


print("\n###########################################################################")
print("#                   Own-CNN Fine Tuning on SVHN dataset                   #")
print("###########################################################################\n")

# Define transform to normalize the data
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the training and test datasets
trainset = torchvision.datasets.SVHN(root='./dataset', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

net = utils.CNN()
net.load_state_dict(torch.load(PATH))
net.cuda(device)  # GPU

log_directory = './logs/cnn_svhn'
utils.train(epochs, trainloader, optim.Adam(net.parameters(), lr=learning_rate), criterion, net, device, log_directory)

PATH = 'models/cnn_svhn.pth'
torch.save(net.state_dict(), PATH)