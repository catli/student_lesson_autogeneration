"""
    Define a simple logistic model on each session data
    Use a neural network with single layer to mimic the output of
    logistic model

    Use the pytorch modules to create this model
"""
import torch
import torch.nn as nn

class Neural_Network(nn.Module):
    def __init__(self, content_dimension, session_len, batch_size):
        super(Neural_Network, self).__init__()

    self.dim_session_content = content_dimension * session_len
    self.batch_size = batch_size
