'''
    Train the model
    for batched data, batch and train
    the approach to training takes a lot of inspiration from how this
    CAHL goal-based model was built https://github.com/CAHLR/goal-based-recommendation
    by WeiJiang (@fabulosa)
'''

from gru import GRU_MODEL as gru_model
import torch
import torch.nn as nn
from process_data import split_train_and_test_data, convert_token_to_matrix
from torch.autograd import Variable
from evaluate import evaluate_loss #, evaluate_precision_and_recall
import torch.utils.data as Data
import numpy as np
import os
import json
import pdb



def train_and_evaluate(model, train_data, val_data,
        optimizer, content_dim, threshold):
    best_vali_loss = None  # set a large number for validation loss at first
    best_vali_accu = 0
    epoch = 0
    training_loss_epoch = []
    eval_loss_epoch = []
    max_epoch = 15 # PLACEHOLDER
    # training data on mini batch
    # [TODO] how to save the training data
    train_keys = np.array([key for key in train_data.keys()])
    train_data_index = torch.IntTensor(range(len(train_data)))
    torch_train_data_index = Data.TensorDataset(train_data_index)
    train_loader = Data.DataLoader(dataset=torch_train_data_index,
                    batch_size=batchsize, shuffle=True,
                    num_workers=2, drop_last=True)
    # validation data on mini batch
    val_keys = np.array([key for key in val_data.keys()])
    val_data_index = torch.IntTensor(range(len(val_data)))
    torch_val_data_index = Data.TensorDataset(val_data_index)
    val_loader = Data.DataLoader(dataset=torch_val_data_index,
                batch_size=batchsize,
                shuffle=True,
                num_workers=2,
                drop_last=True)
    while True:
        epoch += 1
        print('EPOCH %s:' %str(epoch))
        train_loss = train(model, optimizer, train_loader, train_data,
                train_keys, epoch, content_dim)
        training_loss_epoch.append(train_loss)
        print('The average loss of training set for the first %s epochs: %s ' %
                (str(epoch),str(training_loss_epoch)))
        eval_loss, total_predicted, total_label, total_correct = evaluate_loss(
            model, val_loader, val_data, val_keys, content_dim, threshold)
        eval_loss_epoch.append(eval_loss)
        # num_predicted, num_label, num_correct = evaluate_precision_and_recall(
        #     model, val_loader, val_data, val_keys, batchsize, content_dim, threshold)
        # [TODO] write precision and recall to output file
        print('Epoch test: %d / %d  precision and %d / %d  recall' % (
                total_correct, total_predicted,
                total_correct, total_label))

        # print('Epoch test: %d / %d = %f precision and %d / %d = %f recall' % (
        #         total_correct, total_predicted, total_correct/total_predicted,
        #         total_correct, total_label, total_correct/total_label))
        if epoch >= max_epoch:
            # [TODO] consider adding an early stopping logic
            break
    # [TODO] once training complete write_prediction_sample


def train(model, optimizer, loader, train_data,
    train_keys, epoch, content_dim):
    # set in training node
    model.train()
    train_loss = []
    for step, batch_x in enumerate(loader):  # batch_x: index of batch data
        print('Epoch: ', epoch, ' | Iteration: ', step+1)
        # convert token data to matrix
        # need to convert batch_x from tensor flow object to numpy array
        # before converting to matrix
        input_padded, label_padded, seq_len = convert_token_to_matrix(
            batch_x[0].numpy(), train_data, train_keys, content_dim)
        # Variable, used to set tensor, but no longer necessary
        # Autograd automatically supports tensor with requires_grade=True
        #  https://pytorch.org/docs/stable/autograd.html?highlight=autograd%20variable
        padded_input = Variable(torch.Tensor(input_padded), requires_grad=False)#.cuda()
        padded_label = Variable(torch.Tensor(label_padded), requires_grad=False)#.cuda()

        # clear gradients and hidden state
        # [TODO] check why you need to init hidden layer
        optimizer.zero_grad()
        #model.hidden = model.init_hidden()
        #model.hidden[0] = model.hidden[0] #.cuda()
        #model.hidden[1] = model.hidden[1]#.cuda()
        # is this equivalent to generating prediction
        # what is the label generated?
        y_pred = model(padded_input)#.cuda()
        loss = model.loss(y_pred, padded_label)#.cuda()
        print('Epoch ' + str(epoch) + ': ' + 'The '+str(step+1)+'-th iteration: loss '+str(loss.data[0])+'\n')
        loss.backward(retain_graph=True)
        optimizer.step()
        # append the loss after converting back to numpy object from tensor
        train_loss.append(loss.data[0].numpy())

    average_loss = np.mean(train_loss)
    return average_loss


if __name__ == '__main__':
    # only consider grade higher than B or not, pass or not pass

    # set hyper parameters
    nb_lstm_units = 1000
    nb_lstm_layers = 1
    batchsize = 2
    learning_rate = 0.001
    test_perc = 0.2
    threshold = 0.5
    exercise_filename = os.path.expanduser(
                '~/sorted_data/khan_problem_token_3only_tiny')
    content_index_filename = 'data/exercise_index_3only'
    train_data, val_data, _, content_dim = split_train_and_test_data(
                exercise_filename, content_index_filename, test_perc)
    model = gru_model(input_dim = content_dim,
        output_dim = content_dim,
        nb_lstm_layers = nb_lstm_layers,
        nb_lstm_units = nb_lstm_units,
        batch_size = batchsize)
    # [TODO] consider whether to include weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_and_evaluate(model, train_data, val_data,
        optimizer, content_dim, threshold)