import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from scaffolding import LossHistory, ProgressPrinter, Preliminary_information, EarlyStopping, CallbackContext, Trainer, ExperimentTrack
import matplotlib.pyplot as plt

history = LossHistory()
callbacks = [
    ProgressPrinter(n=10),
    Preliminary_information(Xtrain_shape=(1, 6), ytrain_shape=(1, 1), Xval_shape=(1, 6), yval_shape=(1, 1)),
    history,
    EarlyStopping(
        patience=20,
        min_delta=1e-4,
        mode="min",
        restore_best_weights=True,
    ),
    ExperimentTrack(filepath="experiment_track.pth"),
]


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        return (preds == y).float().mean().item()

df = sns.load_dataset("titanic")
df.head()
# keep only these 7 features
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
df.to_csv("titanic.csv", index=False)
df = df[df['age'].notnull()]
# missing values
df['age'] = df['age'].fillna(df['age'].median())
df = pd.get_dummies(df, columns=['sex'], drop_first=True, dtype=float) 
df.head()


# ----------------------------
# Split the dataframe
# ----------------------------

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["survived"])


# standard scaler
scaler = StandardScaler()
# apply scaler to the features
continuous = ['age', 'fare']

scaler = StandardScaler()
train_df[continuous] = scaler.fit_transform(train_df[continuous])
test_df[continuous] = scaler.transform(test_df[continuous])

print(train_df.head())
print(test_df.head())


X = torch.tensor(train_df.drop('survived', axis=1).values, dtype=torch.float32)
y = torch.tensor(train_df['survived'].values, dtype=torch.float32)
Xtest = torch.tensor(test_df.drop('survived', axis=1).values, dtype=torch.float32)
ytest = torch.tensor(test_df['survived'].values, dtype=torch.float32)
print(X.shape, y.shape)
y = y.view(-1, 1)
ytest = ytest.view(-1, 1)
print(X.shape, y.shape)
print(Xtest.shape, ytest.shape)

class TitanicModel(torch.nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.fc1 = torch.nn.Linear(6, 16)  # 6 input features, 16 output features
        self.fc2 = torch.nn.Linear(16, 16) # hidden layer with 8 neurons
        self.fc3 = torch.nn.Linear(16, 1)  # 16 input features, 1 output feature
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



model = TitanicModel()
criterion = torch.nn.BCEWithLogitsLoss()  # binary cross entropy loss with logits
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=3e-2)  # Adam optimizer with weight decay


trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    callbacks=callbacks,
)

ctx = trainer.fit(X, y, Xtest, ytest, epochs=10000)

train_acc = accuracy(model, X, y)
val_acc = accuracy(model, Xtest, ytest)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

train_losses = history.train_losses
val_losses = history.val_losses

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()