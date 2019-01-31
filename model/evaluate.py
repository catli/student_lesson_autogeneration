
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torch.utils.data as Data
from process_data import split_train_and_test_data, convert_token_to_matrix
import pdb

# for validation loss in early stopping

def evaluate_loss(model, loader, val_data, val_keys, content_dim, threshold):
    # set in training node
    model.eval()
    val_loss = []
    total_predicted = 0
    total_label = 0
    total_correct = 0
    for step, batch_x in enumerate(loader):  # batch_x: index of batch data
        print('Evaluate Loss | Iteration: ', step+1)
        # convert token data to matrix
        # need to convert batch_x from tensor flow object to numpy array
        # before converting to matrix
        input_padded, label_padded, seq_len = convert_token_to_matrix(
            batch_x[0].numpy(), val_data, val_keys, content_dim)
        # Variable, used to set tensor, but no longer necessary
        # Autograd automatically supports tensor with requires_grade=True
        #  https://pytorch.org/docs/stable/autograd.html?highlight=autograd%20variable
        padded_input = Variable(torch.Tensor(input_padded), requires_grad=False)#.cuda()
        padded_label = Variable(torch.Tensor(label_padded), requires_grad=False)#.cuda()
        # clear gradients and hidden state
        model.hidden = model.init_hidden()
        # model.hidden[0] = model.hidden[0]#.cuda()
        # model.hidden[1] = model.hidden[1]#.cuda()
        # is this equivalent to generating prediction
        # what is the label generated?
        y_pred = model(padded_input)#.cuda()
        loss = model.loss(y_pred, padded_label)#.cuda()
        # append the loss after converting back to numpy object from tensor
        val_loss.append(loss.data.numpy())
        num_predicted, num_label, num_correct = find_correct_predictions(
            y_pred, padded_label, threshold)#.cuda()
    average_loss = np.mean(val_loss)
    total_predicted += num_predicted
    total_label = num_label
    total_correct += num_correct
    return average_loss, total_predicted, total_label, total_correct


# def evaluate_precision_and_recall(model, loader, val_data, val_keys,  batchsize, content_dim,
#     threshold):
#     model.eval()
#     total_correct = 0.0
#     total_predicted = 0.0
#     total_label = 0.0
#     for step, batch_x in enumerate(loader):  # batch_x: index of batch data
#         processed_data = convert_token_to_matrix(
#             batch_x[0].numpy(), val_data, val_keys, content_dim)
#         # depending on
#         padded_input = Variable(torch.Tensor(processed_data[0]), requires_grad=False)#.cuda()
#         padded_label = Variable(torch.Tensor(processed_data[1]), requires_grad=False)#.cuda()
#         seq_len = processed_data[2]

#         # clear hidden states
#         model.hidden = model.init_hidden()
#         # [TODO] update the hidden unit, may only need one
#         # model.hidden[0] = model.hidden[0]#.cuda()
#         # model.hidden[1] = model.hidden[1]#.cuda()
#         # compute output
#         y_pred = model(padded_input)#.cuda()
#         # only compute the loss for testing period
#         # [TODO] padded_input may need to be masked before calculating precision / recall?
#         # [TODO] add write prediction sample
#         num_predicted, num_label, num_correct = find_correct_predictions(
#             y_pred, padded_label, threshold)#.cuda()
#     return num_predicted, num_label, num_correct



# def evaluate_correct_and_print_sample(sample_writer, student,
#         input, output, label, threshold, index_to_content_map, perc_sample_print):
#     '''
#         compare the predicted list and the actual rate
#         then calculate the recall % and the prediction %
#         also print out 
#     '''
#     # create binary vectors whether a learners should work
#     # on an item
#     output_ones, label_ones, correct_prediction = find_correct_predictions(
#                                             output, label, threshold)
#     # write a random sample (i.e. 0.01 = 1% )
#     if random.random()<=perc_sample_print:
#         write_prediction_sample(sample_writer, student,
#                     input, output_ones, label_ones, correct_prediction,
#                     index_to_content_map)
#     num_correct = np.sum(np.array(correct_prediction.detach()))
#     num_predicted = np.sum(np.array(output_ones.detach()))
#     num_label = np.sum(np.array(label_ones.detach()))
#     return num_correct, num_predicted, num_label


def find_correct_predictions(output, label, threshold):
    '''
        compare the predicted list and the actual rate
        then generate the locaation of correct predictions
    '''
    # set entries below threshold to one
    # thresholder = F.threshold(threshold, 0)
    # any predicted values below threshold be set to 0
    threshold_output = F.threshold(output, threshold, 0)
    # find the difference between label and prediction
    # where prediction is incorrect (label is one and
    # threshold output 0), then the difference would be 1
    # [TODO] PDB.set_trace() to check that
    #     operations for 3 dimension matrix works out
    predict_diff = label - threshold_output
    # set_correct_to_one = F.threshold(0.99, 0)
    incorrect_ones = F.threshold(predict_diff, 0.99, 0)
    num_incorrect = len(torch.nonzero(incorrect_ones))
    # all incorrect prediction would be greater than 0
    num_predicted = len(torch.nonzero(threshold_output))
    num_label = len(torch.nonzero(label))
    num_correct = num_label - num_incorrect
    return num_predicted, num_label, num_correct


def write_prediction_sample(sample_writer,
        student, input, output, label, correct, index_to_content_map):
    '''
        print readable prediciton sample
        for input, output, label expect a matrix that's already
        converted to ones where value above threshold set to 1
    '''
    for i, session_input in enumerate(input):
        readable_input = create_readable_list(input[i], index_to_content_map)
        readable_ouput = create_readable_list(output[i], index_to_content_map)
        readable_label = create_readable_list(label[i], index_to_content_map)
        readable_correct = create_readable_list(correct[i], index_to_content_map)
        sample_writer.write(student + ',' +
                str(readable_input) + ',' +
                str(readable_ouput) + ',' +
                str(readable_label) + ',' +
                str(readable_correct) + '\n')

