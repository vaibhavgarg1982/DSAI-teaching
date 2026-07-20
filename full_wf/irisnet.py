import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scaffolding import LossHistory, ProgressPrinter, Preliminary_information, EarlyStopping, CallbackContext, Trainer
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# convert to torch tensors
Xtrain = torch.tensor(Xtrain, dtype=torch.float32) # convert to torch tensors
ytrain = torch.tensor(ytrain, dtype=torch.long) # convert to torch tensors, because pytorch needs to operate on these tensors
Xtest = torch.tensor(Xtest, dtype=torch.float32) # convert to torch tensors
ytest = torch.tensor(ytest, dtype=torch.long) # convert to torch tensors

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
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    callbacks=[
        ProgressPrinter(n=10),
        Preliminary_information(Xtrain_shape=Xtrain.shape, ytrain_shape=ytrain.shape, Xval_shape=Xtest.shape, yval_shape=ytest.shape),
        history,
        EarlyStopping(
            patience=20,
            min_delta=1e-4,
            mode="min",
            restore_best_weights=True,
        ),
    ],
)

ctx = trainer.fit(Xtrain, ytrain, Xtest, ytest, epochs=10000)

train_losses = history.train_losses
val_losses = history.val_losses

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()

# save the model state_dict to a file
torch.save(model.state_dict(), 'iris_model.pth')