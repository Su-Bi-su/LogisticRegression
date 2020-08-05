import torch
import torch.nn as nn
import numpy as np


# Prepare the Data

xy = np.loadtxt('D:\\pyTorch Review\\1st Exercise\\data\\data_diabetes.csv', delimiter=',', dtype=np.float32)

x = torch.from_numpy(xy[:, 0:-1])
y = torch.from_numpy(xy[:, [-1]])


# Build the model


# Training
# 1. Move the data and model to gpu if available
# 2. define optimizer
# 3. Define Loss function
# 4. Then start the training loop







# Evaluate the network


