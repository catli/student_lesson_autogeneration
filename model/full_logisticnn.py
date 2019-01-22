"""
    Define a simple logistic model on each session data
    Use a neural network with single layer to mimic the output of
    logistic model

    Define each step of the model explicity

    from: https://medium.com/dair-ai/a-simple-neural-network-from-scratch-with-pytorch-and-google-colab-c7f3830618e0
"""
import torch
import torch.nn as nn


class Neural_Network(nn.Module):
    def __init__(self, content_dimension, session_len):
        super(Neural_Network, self).__init__()

        # [TODO]: Add a dimenion for student batch?
        # Parameters
        # the input size is the total number of content x 
        # the allowed length of the session
        self.inputSize = content_dimension * session_len
        self.outputSize = content_dimension * session_len
        self.hiddenSize = 1

        # Weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # inputsize X 1 tensor
        # self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor

    def forward(self, X):
        self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorch
        # self.z2 = self.sigmoid(self.z) # activation function
        # self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z) # final activation function
        print('o forward')
        print(o)
        return o

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        # self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        # self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        # self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        # self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
        self.W1 += torch.matmul(torch.t(X), self.o_delta)

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")

    def predict(self, xPredicted):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))