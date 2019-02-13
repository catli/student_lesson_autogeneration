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
# [EMBED TODO]
from sklearn.decomposition import PCA


class GRU_MODEL(nn.Module):
    def __init__(self, input_dim, output_dim, nb_lstm_layers, nb_lstm_units, batch_size, include_correct):
        super(GRU_MODEL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim # num of possible content
        self.nb_lstm_layers = nb_lstm_layers
        self.nb_lstm_units = nb_lstm_units
        self.batch_size = batch_size
        self.model = nn.GRU(input_size=self.input_dim,
                            hidden_size=nb_lstm_units,
                            num_layers=nb_lstm_layers,
                            batch_first=True)
        self.hidden_to_output = nn.Linear(self.nb_lstm_units,
                                          self.output_dim)
        # [EMBED TODO]: Add the inputs for embeddings
        self.embedding = nn.Embedding(
            # [EMBED TODO] if is_correct included, then
            #    double or increase output dim to value
            num_embeddings = output_dim+1, # output_dim = num of possible content
            embedding_dim = self.input_dim, # number of hidden dimensions
            padding_idx = 0
            )

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
        batch_embeddings = self.create_embeddings(batch_data)
        packed_input = utils.rnn.pack_padded_sequence(
            batch_embeddings, seq_lens, batch_first=True)
        # [EMBED TODO]
        # run through the GRU model
        gru_out, _ = self.model(packed_input, self.hidden)
        # undo packing operation
        unpacked_out, _ = utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True)
        # [TODO] check to see what this unpacking is doing
        unpacked_out = unpacked_out.contiguous()
        unpacked_out = unpacked_out.view(-1, unpacked_out.shape[2])
        unpacked_out = self.hidden_to_output(unpacked_out)
        # Add a sigmoid layer
        sigmoid_out = torch.sigmoid(unpacked_out)
        output = sigmoid_out.view(self.batch_size, -1, self.output_dim)
        return output

    def create_embeddings(self, batch_data):
        '''
            batch data will be in the format of long tensors
            with the format [1,2,4,3],
        '''
        # convert to long tensor before feeding
        embeddings = self.embedding(batch_data.long())
        # [EMBED TODO] preview the batch data to check operation
        batch_embeddings =torch.sum(embeddings, dim = 2)
        # set the normalization denominator
        norm =  torch.Tensor(torch.sum(embeddings!=0,dim=2).float())
        norm[norm==0] = 1
        # [EMBED TODO] is it necessary to normalize the input?
        return torch.div(batch_embeddings, norm)

    def print_embeddings(self, content_dim, epoch):
        '''
            plot embeddings for each content array
        '''
        embedding_output = []
        for i in range(content_dim):
            content_embedding = self.embedding(
                torch.LongTensor([[i]])).detach().numpy()[0][0]
            embedding_output.append(content_embedding)
        pca = PCA(n_components=2)
        pca_embedding = pca.fit_transform(embedding_output)
        plot = plt.scatter(pca_embedding[:,0], pca_embedding[:,1], color = 'white')
        for i in range(content_dim):
            plot = plt.text(pca_embedding[i,0], pca_embedding[i,1], i, fontsize=5)
        plot.figure.savefig(os.path.expanduser('~/sorted_data/output/embed_' + str(epoch) +'.jpg'))
        plt.clf()	


    def loss(self, output, label):
        # [TODO] figure out how to improve the loss function
        mse = nn.MSELoss()
        loss = mse(output, label)
        return loss
