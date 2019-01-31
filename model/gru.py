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
        self.hidden = self.init_hidden()
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
        '''
        seq_len = batch_data.size()[1]
        # check that the gru unit treats the s
        gru_out, self.hidden = self.model(batch_data, self.hidden)
        output = self.hidden_to_output(gru_out)
        return output

    def loss(self, output, label):
        # [TODO] test out different loss functions
        # cross_entropy = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        loss = mse(output, label)
        return loss
