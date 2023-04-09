import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self, activation):
        super().__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        # Activation Layer
        self.activation = nn.LeakyReLU(negative_slope=.1) if activation == "LeakyReLu" else nn.Tanh()

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def train(epochs, trainloader, optimizer, criterion, net, device, directory):
    writer = SummaryWriter(directory)

    # Train the network
    for epoch in range(epochs):  # Loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; dataset is a list of [inputs, labels]
            inputs, labels = data

            inputs = torch.as_tensor(inputs).to(device)
            labels = torch.as_tensor(labels).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net.forward(inputs)  # GPU
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(f'[{epoch + 1}] loss: {running_loss / 390:.3f}')

    writer.flush()
    writer.close()


def test(testloader, net, device):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images = torch.as_tensor(images).to(device)
            labels = torch.as_tensor(labels).to(device)

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images = torch.as_tensor(images).to(device)
            labels = torch.as_tensor(labels).to(device)

            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

