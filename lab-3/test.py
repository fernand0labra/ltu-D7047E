import torch
import torchvision
import torchvision.transforms as transforms
import utils.model_utils as model_utils
import utils.visualization_utils as visualization_utils

# Hyperparameters
batch_size = 128

# Define execution device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Define transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the training and test datasets
testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
images, labels = next(iter(testloader))
images = images.to(device)  # GPU

print("\n###########################################################################")
print("#                   CNN Small Training on MNIST dataset                   #")
print("###########################################################################\n")

PATH = 'models/cnn_mnist_small.pth'
net = model_utils.CNN()
net.load_state_dict(torch.load(PATH))
net.to(device)  # GPU

# model_utils.test(testloader, net, device, classes)
features = model_utils.extract_features(batch_size, images, net)

# visualization_utils.visualize_pca(features, images.cpu(), labels, scatter_images=True, img_limit=0.4)
visualization_utils.visualize_tsne(features, images.cpu(), labels, scatter_images=False, img_limit=2)
# visualization_utils.visualize_filters_activations(images[0], net)

print("\n###########################################################################")
print("#                  CNN Medium Training on MNIST dataset                   #")
print("###########################################################################\n")

PATH = 'models/cnn_mnist_medium.pth'
net.load_state_dict(torch.load(PATH))
net.to(device)  # GPU

# model_utils.test(testloader, net, device, classes)
features = model_utils.extract_features(batch_size, images, net)

# visualization_utils.visualize_pca(features, images.cpu(), labels, scatter_images=True, img_limit=2)
visualization_utils.visualize_tsne(features, images.cpu(), labels, scatter_images=False, img_limit=2)
# visualization_utils.visualize_filters_activations(images[0], net)
