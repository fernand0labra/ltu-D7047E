import torch
import torchvision
import torchvision.transforms as transforms
import utils
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 512

# Define execution device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n###########################################################################")
print("#                   CNN Small Training on MNIST dataset                   #")
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

PATH = 'models/cnn_mnist_small.pth'
net = utils.CNN()
net.load_state_dict(torch.load(PATH))
net.cuda(device)  # GPU

image = torch.as_tensor(next(iter(testloader))[0][0]).to(device)
# plt.imshow(image[0].cpu().numpy(), cmap='gray')
# plt.show()

conv1_feature_maps = net.maxpool(net.relu(net.conv1(image)))

conv1_flattened_features = conv1_feature_maps\
    .view(16, 14 * 14).cpu().detach().numpy()
conv2_flattened_features = net.maxpool(net.relu(net.conv2(conv1_feature_maps)))\
    .view(32, 7 * 7).cpu().detach().numpy()
features = [conv2_flattened_features]

utils.visualize_pca(features)
utils.visualize_tsne(features)
# utils.visualize_filters_activations([conv1_flattened_features, conv2_flattened_features])
utils.test(testloader, net, device, classes)

print("\n###########################################################################")
print("#                  CNN Medium Training on MNIST dataset                   #")
print("###########################################################################\n")

PATH = 'models/cnn_mnist_medium.pth'
net.load_state_dict(torch.load(PATH))
net.cuda(device)  # GPU

conv1_feature_maps = net.maxpool(net.relu(net.conv1(image)))

conv1_flattened_features = conv1_feature_maps\
    .view(16, 14 * 14).cpu().detach().numpy()
conv2_flattened_features = net.maxpool(net.relu(net.conv2(conv1_feature_maps)))\
    .view(32, 7 * 7).cpu().detach().numpy()
features = [conv2_flattened_features]

utils.visualize_pca(features)
utils.visualize_tsne(features)
# utils.visualize_filters_activations(image, net)
utils.test(testloader, net, device, classes)
