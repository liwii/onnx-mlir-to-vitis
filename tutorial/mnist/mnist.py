import torch
from torch import nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms

class MnistNetwork(nn.Module):
    def __init__(self):
        super(MnistNetwork, self).__init__()
        self.layer1 = nn.Linear(784, 20)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(20, 20)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        return x
if __name__ == '__main__':

    model = MnistNetwork()

    transform=transforms.Compose([
        transforms.ToTensor()
        ])

    train = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)

    test = datasets.MNIST('../data', train=False,
                        transform=transform)
    test_images = []
    test_labels = []
    for img, label in test:
        test_images.append(img.detach().numpy().reshape((28*28,)))
        test_labels.append(label)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):
        running_loss = 0.0
        for data in trainset:
            X, y = data
            model.zero_grad()
            output = model(X.view(-1, 28*28))
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch: {epoch}, Loss: {loss}")

    correct = 0
    total = 0

    with torch.no_grad():
        for data in trainset:
                X, y = data
                output = model(X.view(-1, 784))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total +=1

    print("Training Accuracy: ", round(correct/total, 3))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testset:
                X, y = data
                output = model(X.view(-1, 784))
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total +=1

    print("Test Accuracy: ", round(correct/total, 3))

    with torch.no_grad():
        torch.onnx.export(model, torch.tensor(test_images[0]), './tutorial/mnist/mnist.onnx')