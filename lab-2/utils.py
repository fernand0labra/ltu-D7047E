import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class AlexNet(nn.Module):
    def __init__(self, mode):
        super().__init__()

        # AlexNet Model
        self.alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=None, force_reload=True) \
            if mode == "fine_tuning" else torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', force_reload=True)

        # Alter last linear layer to output on the 10 class CIFAR10 dataset
        self.alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)

        if mode == "feature_extraction":  # Freeze all layers except last linear one
            for param in self.alexnet.parameters():
                param.requires_grad = False

            for param in self.alexnet.classifier[6].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.alexnet(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=7 * 7 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 32)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
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

        print(f'[{epoch + 1}] loss: {running_loss / 469:.3f}')

    writer.flush()
    writer.close()


def test(testloader, net, device, classes):
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

