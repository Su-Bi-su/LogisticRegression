import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
y = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])


# Build the model
class SimpleBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.z = nn.Linear(1, 1)

    def forward(self, input_data):
        prediction = F.sigmoid(self.z(input_data))
        return prediction


# Move model and data to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = SimpleBinaryClassifier().to(device)
x = x.to(device)
y = y.to(device)


# Define optimizer and Loss function

optimizer = optim.SGD(model.parameters(), lr=0.001)
lossfn = nn.BCELoss()

train_loss = []
# train the model
for epoch in range(1000):

    prediction = model(x)
    loss = lossfn(prediction, y)

    print(epoch, loss.item())
    train_loss.append(loss)

    #zero gradient, perform backprop and update the parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# After Training
#plot loss curve

fig, ax = plt.subplots()
ax.plot(range(1000), train_loss)
ax.set_title("Loss Curve")
plt.xlabel("#epoches")
plt.ylabel("loss")
plt.show()

print("After Training:")

x_new = torch.tensor([[25.0]]).to(device)

with torch.no_grad():
    y_hat_new = model(x_new)

    output = 1 if y_hat_new.item() >= 0.5 else 0

    print(x_new.item(), "---------------->", output)








