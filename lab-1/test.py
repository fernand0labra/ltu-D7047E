import torch
import torch.nn as nn
import utils
import torchvision
import torchvision.transforms as transforms


# Hyperparameters
batch_size = 4

# Load and normalize CIFAR10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define loss criterion
criterion = nn.CrossEntropyLoss

# Define execution device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nTESTING PHASE\n")

print("\n###########################################################################")
print("# CNN with LeakyReLu activation and Stochastic Gradient Descent optimizer #")
print("###########################################################################\n")

PATH = './models/cifar_sgd_leaky_relu.pth'
net = utils.Net(activation="LeakyReLu")
net.load_state_dict(torch.load(PATH))
net.cuda(device)  # GPU

utils.test(testloader, net, device)

print("\n###########################################################################")
print("#           CNN with LeakyReLu activation and Adam optimizer              #")
print("###########################################################################\n")

PATH = './models/cifar_adam_leaky_relu.pth'
net = utils.Net(activation="LeakyReLu")
net.load_state_dict(torch.load(PATH))
net.cuda(device)  # GPU

utils.test(testloader, net, device)

print("\n###########################################################################")
print("#              CNN with Tanh activation and Adam optimizer                #")
print("###########################################################################\n")

PATH = './models/cifar_adam_tanh.pth'
net = utils.Net(activation="Tanh")
net.load_state_dict(torch.load(PATH))
net.cuda(device)  # GPU

utils.test(testloader, net, device)
