
import torch
import torch.nn as nn
from logisticnn import Neural_Network
import pdb


X = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float) # 3 X 2 tensor
y = torch.tensor(([92], [100], [89]), dtype=torch.float) # 3 X 1 tensor
xPredicted = torch.tensor(([33.1632, 99.7444]), dtype=torch.float) # 1 X 2 tensor


content_dimenion =2
session_len =1
NN = Neural_Network(content_dimenion, session_len)
for i in range(1000):  # trains the NN 1,000 times
    print ("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
NN.saveWeights(NN)
# NN.predict()
pdb.set_trace()