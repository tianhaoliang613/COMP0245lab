import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Data Preparation
# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset
])

# Download and load the training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transform, download=True)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 2. Model Construction
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 32)  # Input layer to hidden layer
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 10)     # Hidden layer to output layer
        self.softmax = nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

# Deep Learning Model
class d_MLP(nn.Module):
    def __init__(self):
        super(d_MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(28*28, 30)  # Input layer to hidden layer
        self.hidden2 = nn.Linear(30, 46)      # First hidden layer to second hidden layer
        self.output = nn.Linear(46, 10)       # Second hidden layer to output layer
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        x = self.softmax(x)
        return x

class dd_MLP(nn.Module):
    def __init__(self):
        super(dd_MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(28*28, 30)  # Input layer to hidden layer
        self.hidden2 = nn.Linear(30, 24)      # First hidden layer to second hidden layer
        self.hidden3 = nn.Linear(24, 16)      # Second hidden layer to third hidden layer
        self.hidden4 = nn.Linear(16, 16)      # Third hidden layer to fourth hidden layer
        self.output = nn.Linear(16, 10)       # Fourth hidden layer to output layer
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.relu(self.hidden4(x))
        x = self.output(x)
        x = self.softmax(x)
        return x

model = dd_MLP()

# 3. Model Compilation
criterion = nn.NLLLoss()  # Negative Log Likelihood Loss (used with LogSoftmax)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Model Training
epochs = 20
train_losses = []
valid_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # target is not one-hot encoded in PyTorch
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_losses.append(epoch_loss / len(train_loader))
    train_accuracy = 100. * correct / len(train_loader.dataset)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy:.2f}%')

# 5. Model Evaluation
model.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100. * correct / len(test_loader.dataset)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')