import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# dataset preparation
full_train_dataset = datasets.MNIST(root='./data', train=True,
                                    transform=transform, download=True)
train_size = int(0.80 * len(full_train_dataset))  # ~50000
val_size = len(full_train_dataset) - train_size   # ~10000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))

test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transform, download=True)

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.hidden(x))
        x = self.softmax(self.output(x))
        return x

# Training and validation function
def train_and_validate(batch_size, lr, epochs=20):
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLP()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss, correct = 0, 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        print(f"[Batch={batch_size}, LR={lr}] Epoch {epoch+1}/{epochs} | "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    return best_val_acc, train_losses, val_losses, best_model_state

# Hyperparameter tuning
batch_sizes = [16, 32, 64]
learning_rates = [0.001, 0.01, 0.1]
results = {}

for bs in batch_sizes:
    for lr in learning_rates:
        val_acc, train_losses, val_losses, model_state = train_and_validate(bs, lr)
        results[(bs, lr)] = {
            'val_acc': val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_state': model_state
        }



#  Train Loss 
plt.figure(figsize=(10,6))
for (bs, lr), res in results.items():
    plt.plot(res['train_losses'], label=f'Train Loss (BS={bs}, LR={lr})')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss Curves for Different Hyperparameters')
plt.legend()
plt.show()

#  Validation Loss 
plt.figure(figsize=(10,6))
for (bs, lr), res in results.items():
    plt.plot(res['val_losses'], label=f'Val Loss (BS={bs}, LR={lr})')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Curves for Different Hyperparameters')
plt.legend()
plt.show()


# Evaluate best model on test set
best_params = max(results.items(), key=lambda x: x[1]['val_acc'])
best_bs, best_lr = best_params[0]
best_model_state = best_params[1]['model_state']
print(f"\nBest Model: Batch Size={best_bs}, LR={best_lr}, Val Acc={best_params[1]['val_acc']:.2f}%")


best_model = MLP()
best_model.load_state_dict(best_model_state)
best_model.eval()

test_loader = DataLoader(test_dataset, batch_size=best_bs, shuffle=False)
# Get predictions
all_preds, all_targets = [], []
with torch.no_grad():
    for data, target in test_loader:
        output = best_model(data)
        pred = output.argmax(dim=1)
        all_preds.extend(pred.numpy())
        all_targets.extend(target.numpy())

# Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Best Model)')
plt.show()

# Visualize misclassified samples
test_images = test_dataset.data
test_labels = test_dataset.targets

mis_idx = np.where(np.array(all_preds) != np.array(all_targets))[0]
print(f"Number of misclassified samples: {len(mis_idx)}")

# Plot some misclassified examples
plt.figure(figsize=(8,8))
for i, idx in enumerate(mis_idx[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_images[idx], cmap='gray')
    plt.title(f"True: {test_labels[idx].item()}, Pred: {all_preds[idx]}")
    plt.axis('off')
plt.suptitle('Examples Where Model Failed')
plt.show()
