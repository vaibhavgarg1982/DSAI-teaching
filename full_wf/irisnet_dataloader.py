import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from scaffolding import DataLoaderTrainer, EarlyStopping, LossHistory, Preliminary_information, ProgressPrinter


data = load_iris()
X = data.data
y = data.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

Xtrain = torch.tensor(Xtrain, dtype=torch.float32)
ytrain = torch.tensor(ytrain, dtype=torch.long)
Xtest = torch.tensor(Xtest, dtype=torch.float32)
ytest = torch.tensor(ytest, dtype=torch.long)

train_dataset = TensorDataset(Xtrain, ytrain)
val_dataset = TensorDataset(Xtest, ytest)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


class IrisNet(torch.nn.Module):
    '''IrisNet is a simple feedforward neural network for classifying the Iris dataset.
    It consists of three fully connected layers with ReLU activation functions.
    This inherits from torch.nn.Module, which is the base class for all neural network modules in PyTorch.
    '''

    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = torch.nn.Linear(4, 10)
        self.fc2 = torch.nn.Linear(10, 5)
        self.fc3 = torch.nn.Linear(5, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


model = IrisNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

history = LossHistory()
trainer = DataLoaderTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    callbacks=[
        ProgressPrinter(n=10),
        Preliminary_information(
            Xtrain_shape=Xtrain.shape,
            ytrain_shape=ytrain.shape,
            Xval_shape=Xtest.shape,
            yval_shape=ytest.shape,
        ),
        history,
        EarlyStopping(
            patience=20,
            min_delta=1e-4,
            mode="min",
            restore_best_weights=True,
        ),
    ],
)

ctx = trainer.fit(train_loader, val_loader, epochs=10000)

train_losses = history.train_losses
val_losses = history.val_losses

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'iris_model_dataloader.pth')

def calc_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

train_acc = calc_accuracy(model, train_loader)
val_acc = calc_accuracy(model, val_loader)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")