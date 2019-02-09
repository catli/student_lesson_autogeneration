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
from process_data import split_train_and_test_data, convert_token_to_matrix, extract_content_map
from torch.autograd import Variable
from evaluate import evaluate_loss  # , evaluate_precision_and_recall
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pdb
import yaml


def train_and_evaluate(model, full_data, train_keys, val_keys,
                       optimizer, content_dim, threshold,
                        max_epoch, file_affix, include_correct):
# output_sample_filename, exercise_to_index_map, , perc_sample_print
    result_writer = open(
        os.path.expanduser('~/sorted_data/output_%s' % file_affix), 'w')
    best_vali_loss = None  # set a large number for validation loss at first
    best_vali_accu = 0
    epoch = 0
    training_loss_epoch = []
    eval_loss_epoch = []
    # training data on mini batch
    train_data_index = torch.IntTensor(range(len(train_keys)))
    torch_train_data_index = Data.TensorDataset(train_data_index)
    train_loader = Data.DataLoader(dataset=torch_train_data_index,
                                   batch_size=batchsize,
                                   num_workers=2,
                                   drop_last=True)
    # validation data on mini batch
    val_data_index = torch.IntTensor(range(len(val_keys)))
    torch_val_data_index = Data.TensorDataset(val_data_index)
    val_loader = Data.DataLoader(dataset=torch_val_data_index,
                                 batch_size=batchsize,
                                 num_workers=2,
                                 drop_last=True)
    for count in range(max_epoch):
        epoch = count + 1
        print('EPOCH %s:' % str(epoch))
        train_loss = train(model, optimizer, full_data, train_loader,
                           train_keys, epoch, content_dim, include_correct)
        training_loss_epoch.append(train_loss)
        print('The average loss of training set for the first %s epochs: %s ' %
              (str(epoch), str(training_loss_epoch)))
        eval_loss, total_predicted, total_label, total_correct, \
            total_no_predicted, total_sessions = evaluate_loss(model, full_data,
                val_loader, val_keys, content_dim, threshold, include_correct)
        eval_loss_epoch.append(eval_loss)
        epoch_result = 'Epoch %d test: %d / %d  precision \
                    and %d / %d  recall with %d / %d no prediction sess \n' % (
            epoch, total_correct, total_predicted,
            total_correct, total_label, total_no_predicted, total_sessions)
        result_writer.write(epoch_result)
        print(epoch_result)
    # plot loss
    plot_loss(training_loss_epoch, file_affix)


def train(model, optimizer, train_data, loader,
          train_keys, epoch, content_dim, include_correct):
    # set in training node
    model.train()
    train_loss = []
    for step, batch_x in enumerate(loader):  # batch_x: index of batch data
        print('Epoch: ', epoch, ' | Iteration: ', step+1)
        # convert token data to matrix
        # need to convert batch_x from tensor flow object to numpy array
        # before converting to matrix
        input_padded, label_padded, seq_lens = convert_token_to_matrix(
            batch_x[0].numpy(), train_data, train_keys, content_dim, include_correct)
        # Variable, used to set tensor, but no longer necessary
        # Autograd automatically supports tensor with requires_grade=True
        #  https://pytorch.org/docs/stable/autograd.html?highlight=autograd%20variable
        padded_input = Variable(torch.Tensor(
            input_padded), requires_grad=False)  # .cuda()
        padded_label = Variable(torch.Tensor(
            label_padded), requires_grad=False)  # .cuda()

        # clear gradients and hidden state
        optimizer.zero_grad()
        model.hidden = model.init_hidden()
        # is this equivalent to generating prediction
        # what is the label generated?
        y_pred = model(padded_input, seq_lens)  # .cuda()
        loss = model.loss(y_pred, padded_label)  # .cuda()
        print('Epoch %s: The %s-th iteration: %s loss \n' %(
            str(epoch), str(step+1), str(loss.data[0].numpy())))
        loss.backward()
        optimizer.step()
        # append the loss after converting back to numpy object from tensor
        train_loss.append(loss.data[0].numpy())
    average_loss = np.mean(train_loss)
    return average_loss


def plot_loss(loss_trend, file_affix):
     # visualize the loss
    write_filename = os.path.expanduser('~/sorted_data/output/loss_plot_%s.jpg'
                                        %file_affix)
    plt.plot(range(len(loss_trend)), loss_trend, 'r--')
    plt.savefig(write_filename)


def run_train_and_evaluate(loaded_params):
    exercise_filename = os.path.expanduser(
        loaded_params['exercise_filename'])
    # output_sample_filename = os.path.expanduser(
    #     loaded_params['output_sample_filename'])
    content_index_filename = loaded_params['content_index_filename']
    # creat ethe filename
    file_affix = 'unit%slayer%sbsize%sthresh%s_%s' % (
        str(nb_lstm_units), str(nb_lstm_layers),
        str(batchsize).replace('.', ''),
        str(threshold).replace('.', ''), str(data_name) )

    train_keys, val_keys, full_data = split_train_and_test_data(
        exercise_filename, content_index_filename, test_perc)
    exercise_to_index_map, content_dim = extract_content_map(
        content_index_filename)
    # if include perc correct in the input, then double dimensions
    if include_correct:
        input_dim = content_dim*2
    else:
        input_dim = content_dim

    model = gru_model(input_dim=input_dim,
                      output_dim=content_dim,
                      nb_lstm_layers=nb_lstm_layers,
                      nb_lstm_units=nb_lstm_units,
                      batch_size=batchsize)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_and_evaluate(model, full_data, train_keys, val_keys,
                       optimizer, content_dim, threshold, max_epoch,
                       file_affix,include_correct)
    torch.save(model.state_dict(), 'output/model_%s' % file_affix)


if __name__ == '__main__':
    # set hyper parameters
    loaded_params = yaml.load(open('input/model_params.yaml', 'r'))
    max_epoch = loaded_params['max_epoch']
    nb_lstm_units = loaded_params['nb_lstm_units']
    nb_lstm_layers = loaded_params['nb_lstm_layers']
    batchsize = loaded_params['batchsize']
    learning_rate = loaded_params['learning_rate']
    test_perc = loaded_params['test_perc']
    threshold = loaded_params['threshold']
    data_name = loaded_params['data_name']
    # perc_sample_print = loaded_params['perc_sample_print']
    include_correct = loaded_params['include_correct']
    run_train_and_evaluate(loaded_params)
