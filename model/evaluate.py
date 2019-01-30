
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torch.utils.data as Data
from data_process import process_data, get_data_from_condense_seq

# for validation loss in early stopping

def evaluate_loss(model, loader, val_data, content_dim):
    # set in training node
    model.train()
    val_loss = []
    for step, (batch_x, batch_y) in enumerate(loader):  # batch_x: index of batch data
        print('Epoch: ', epoch, ' | Iteration: ', step+1)
        input_padded, label_padded, seq_len = convert_token_to_matrix(
            batch_x.numpy(), val_data, content_dim)
        # Variable, used to set tensor, but no longer necessary
        # Autograd automatically supports tensor with requires_grade=True
        #  https://pytorch.org/docs/stable/autograd.html?highlight=autograd%20variable
        padded_input = tensor(torch.Tensor(input_padded), requires_grad=False).cuda()
        padded_label = tensor(torch.Tensor(label_padded), requires_grad=False).cuda()

        # clear gradients and hidden state
        model.hidden = model.init_hidden()
        model.hidden[0] = model.hidden[0].cuda()
        model.hidden[1] = model.hidden[1].cuda()
        # is this equivalent to generating prediction
        # what is the label generated?
        y_pred = model(padded_input, seq_len).cuda()
        loss = model.loss(y_pred, padded_label).cuda()
        val_loss.append(loss.data[0])

    average_loss = np.mean(val_loss)
    return average_loss


def evaluate_precision_and_recall(model, loader, val_data, batchsize, content_dim,
    threshold):

    model.eval()
    total_correct = 0.0
    total_predicted = 0.0
    total_label = 0.0
    for step, (batch_x, batch_y) in enumerate(loader):  # batch_x: index of batch data
        processed_data = convert_token_to_matrix(batch_x.numpy(), val_data, batchsize, content_dim)
        # depending on
        padded_input = Variable(torch.Tensor(processed_data[0]), requires_grad=False).cuda()
        padded_label = Variable(torch.Tensor(processed_data[1]), requires_grad=False).cuda()
        seq_len = processed_data[2]

        # clear hidden states
        model.hidden = model.init_hidden()
        # [TODO] update the hidden unit, may only need one
        model.hidden[0] = model.hidden[0].cuda()
        model.hidden[1] = model.hidden[1].cuda()
        # compute output
        y_pred = model(padded_input, seq_len).cuda()
        # only compute the loss for testing period
        # [TODO] padded_input may need to be masked before calculating precision / recall?
        # [TODO] add write prediction sample
        num_predicted, num_label, num_correct = find_correct_predictions(
            y_pred, label, threshold).cuda()
    return total_correct, total_predicted, total_label



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
    thresholder = torch.nn.Threshold(threshold, 0)
    # any predicted values below threshold be set to 0
    threshold_output = thresholder(output)
    # find the difference between label and prediction
    # where the label is one and threshold output
    # predicts 0, then the difference would be 1
    # [TODO] PDB.set_trace() to check that
    #     operations for 3 dimension matrix works out
    predict_diff = label - threshold_output
    set_correct_to_one = torch.nn.Threshold(0.99, 0)
    num_incorrect = len(set_correct_to_one(predict_diff))
    # all incorrect prediction would be greater than 0
    num_predicted = len(len(torch.nonzero(threshold_output)))
    num_label = torch.nonzero(label)
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

