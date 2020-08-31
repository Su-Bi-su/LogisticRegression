import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Prepare the Data

xy = np.loadtxt('..\\data\\data_diabetes.csv', delimiter=',', dtype=np.float32)

x = torch.from_numpy(xy[0:-1, 0:-1])
y = torch.from_numpy(xy[0:-1, [-1]])


# Build the model

class DiabetesClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 6)
        self.layer2 = nn.Linear(6, 4)
        self.layer3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        layer1_output = self.sigmoid(self.layer1(input))
        layer2_output = self.sigmoid(self.layer2(layer1_output))
        layer3_output = self.sigmoid(self.layer3(layer2_output))

        return layer3_output


# Training
# 1. Move the data and model to gpu if available
# 2. define optimizer
# 3. Define Loss function
# 4. Then start the training loop

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DiabetesClassifier().to(device)
x = x.to(device)
y = y.to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_loss = []
for epoch in range(1500):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + loss + backprop + updatte
    prediction = model(x)
    loss = criterion(prediction, y)

    print(epoch, loss.item())
    train_loss.append(loss)

    loss.backward()
    optimizer.step()

# After Training
# plot loss curve

fig, ax = plt.subplots()
ax.plot(range(1500), train_loss)
ax.set_title("Loss Curve")
plt.xlabel("#epoches")
plt.ylabel("loss")
plt.show()

# Evaluate the network

print("After Training:")

x_new = torch.from_numpy(xy[[-1], 0:-1]).to(device)

with torch.no_grad():
    y_hat_new = model(x_new)

    output = 1 if y_hat_new.item() >= 0.5 else 0

    print("Output for new data is : ", output)

    print("That means for new data prediction is ---------------->", "Tested Positive" if output == 1 else "Tested Negative")
