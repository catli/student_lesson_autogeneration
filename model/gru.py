"""
    Define the RNN LSTM/GRU model to predict student session activities
    The model will input a sequence of sessions for a learner
    and predict future ones
"""


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.tensor as tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from torch.utils.data import sampler
import random
import pdb
import time



class GRU_MODEL(nn.Module):
    def __init__(self, input_dim, output_dim, nb_lstm_layers, nb_lstm_units, batch_size):
        super(GRU_MODEL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_lstm_layers = nb_lstm_layers
        self.nb_lstm_units = nb_lstm_units
        self.batch_size = batch_size
        self.model = nn.GRU(input_size=self.input_dim, 
                hidden_size=nb_lstm_units,
                num_layers=nb_lstm_layers,
                batch_first=True)
        self.hidden_to_output = nn.Linear(self.nb_lstm_units,
            self.output_dim)


    def init_hidden(self):
        '''
            initiate hidden layer as tensor
            approach for pytorch 1.0.0
            earlier versions use "Variable" to initiate tensor
            variable
        '''
        self.hidden_layer = tensor(torch.zeros(self.nb_lstm_layers,
                    self.batch_size, self.nb_lstm_units))

    def forward(self, batch_data):
        '''
            apply the forward function for the model
            clarifying with example from:
            towardsdatascience.com/\
            taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        '''
        self.hidden = self.init_hidden()
        batch_size, seq_len, _ = batch_data.size()
        # check that the gru unit treats the s
        packed_input = utils.rnn.pack_padded_sequence(
            batch_data, seq_len, batch_first=True)
        gru_out, self.hidden = self.model(batch_data, self.hidden)
        unpacked_out = utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # [TODO] check to see what this unpacking is doing
        unpacked_out = unpacked_out.contiguous()
        unpacked_out = X.view(-1, unpacked_out.shape[2])
        output = self.hidden_to_output(gru_out)
        # Add a sigmoid layer
        sigmoid_out = F.sigmoid(self.linear(output))
        output = sigmoid_out.view(batch_size, seq_len, self.output_dim)

        return output

    def loss(self, output, label):
        # [TODO] figure out how to structure the loss function
        # cross_entropy = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        loss = mse(output, label)
        return loss

