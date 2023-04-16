import torch
import torch.nn as nn
import scipy
import utils
import torchvision
import torchvision.transforms as transforms

# Hyperparameters
batch_size = 128

# Load and normalize CIFAR10
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define loss criterion
criterion = nn.CrossEntropyLoss

# Define execution device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nTESTING PHASE\n")

print("\n###########################################################################")
print("#                  AlexNet Fine Tuning on CIFAR-10 dataset                #")
print("###########################################################################\n")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

PATH = 'models/alexnet_fine_tuning_cifar10.pth'
net = utils.AlexNet("fine_tuning")
net.load_state_dict(torch.load(PATH))
net.cuda(device)  # GPU

utils.test(testloader, net, device, classes)

print("\n###########################################################################")
print("#            AlexNet Feature Extraction on CIFAR-10 dataset               #")
print("###########################################################################\n")

PATH = 'models/alexnet_feature_extraction_cifar10.pth'
net = utils.AlexNet("feature_extraction")
net.load_state_dict(torch.load(PATH))
net.cuda(device)  # GPU

utils.test(testloader, net, device, classes)

print("\n###########################################################################")
print("#                      Own-CNN Training on MNIST dataset                  #")
print("###########################################################################\n")

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Define transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the training and test datasets
testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

PATH = 'models/cnn_mnist.pth'
net = utils.CNN()
net.load_state_dict(torch.load(PATH))
net.cuda(device)  # GPU

utils.test(testloader, net, device, classes)

print("\n###########################################################################")
print("#            Own-CNN Feature Extraction from MNIST to SVHN datasets       #")
print("###########################################################################\n")

# Define transform to normalize the data
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the training and test datasets
testset = torchvision.datasets.SVHN(root='./dataset', split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# http://ufldl.stanford.edu/housenumbers/
utils.test(testloader, net, device, classes)
