"""
    Define the RNN LSTM/GRU model to predict student session activities
    The model will input a sequence of sessions for a learner
    and predict future ones
"""


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from torch.utils.data import sampler
import torch.nn.utils as utils
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
        self.init_hidden()
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
        self.hidden = Variable(torch.zeros(self.nb_lstm_layers,
                                                 self.batch_size, self.nb_lstm_units))

    def forward(self, batch_data, seq_lens):
        '''
            apply the forward function for the model
            clarifying with example from:
            towardsdatascience.com/\
            taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        '''
        # pack_padded_sequence so that
        # padded items in the sequence won't be shown
        packed_input = utils.rnn.pack_padded_sequence(
            batch_data, seq_lens, batch_first=True)
        # run through the GRU model
        gru_out, self.hidden = self.model(packed_input, self.hidden)
        # undo packing operation
        unpacked_out, _ = utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True)
        unpacked_out = unpacked_out.contiguous()
        unpacked_out = unpacked_out.view(-1, unpacked_out.shape[2])
        unpacked_out = self.hidden_to_output(unpacked_out)
        # add a sigmoid layer
        sigmoid_out = torch.sigmoid(unpacked_out)
        output = sigmoid_out.view(self.batch_size, -1, self.output_dim)
        return output

    def loss(self, output, label):
        # use mean-square error loss function
        # tested cross-entropy, which did not work as well
        mse = nn.MSELoss()
        loss = mse(output, label)
        return loss
