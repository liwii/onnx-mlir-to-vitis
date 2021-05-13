import torch
from torch import nn
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class IrisNetwork(nn.Module):
    def __init__(self):
        super(IrisNetwork, self).__init__()
        self.layer1 = nn.Linear(4, 10)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(10, 10)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(10, 3)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x

model = IrisNetwork()
iris = datasets.load_iris()

train_X, test_X, train_Y, test_Y = train_test_split(iris.data, iris.target, test_size=0.2)

train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y)

test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y)


criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):
    running_loss = 0.0
    for i in range(len(train_X)):
        row = train_X[i]
        label = train_Y[i:i+1]
        optimizer.zero_grad()
        output = model(row)
        output = output.unsqueeze(0)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch: {epoch}, Loss: {loss}")

correct = 0
with torch.no_grad():
    for i in range(len(test_X)):
        row = test_X[i]
        label = test_Y[i].item()
        prediction = np.argmax(np.array(model(row)))
        if prediction == label:
            correct += 1

print(f"Accuracy: {correct / len(test_Y)}")


with torch.no_grad():
    torch.onnx.export(model, train_X[0], './tutorial/iris/iris.onnx')
