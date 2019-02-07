
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data as Data
import numpy as np
from process_data import split_train_and_test_data, convert_token_to_matrix
import random
import pdb

# for validation loss in early stopping


def evaluate_loss(model, val_data, loader, val_keys, content_dim, threshold,
                  output_sample_filename, epoch, exercise_to_index_map, 
                  perc_sample_print, include_correct):
    # set in training node
    # perc_sample_print = 0.05 # set the percent sample

    model.eval()
    val_loss = []
    total_predicted = 0
    total_label = 0
    total_correct = 0
    total_no_predicted = 0
    total_sessions = 0
    for step, batch_x in enumerate(loader):  # batch_x: index of batch data
        print('Evaluate Loss | Iteration: ', step+1)
        # convert token data to matrix
        # need to convert batch_x from tensor flow object to numpy array
        # before converting to matrix
        input_padded, label_padded, seq_lens = convert_token_to_matrix(
            batch_x[0].numpy(), val_data, val_keys, content_dim, include_correct)
        # Variable, used to set tensor, but no longer necessary
        # Autograd automatically supports tensor with requires_grade=True
        #  https://pytorch.org/docs/stable/autograd.html?highlight=autograd%20variable
        padded_input = Variable(torch.Tensor(
            input_padded), requires_grad=False)  # .cuda()
        padded_label = Variable(torch.Tensor(
            label_padded), requires_grad=False)  # .cuda()
        # clear gradients and hidden state
        model.hidden = model.init_hidden()
        # is this equivalent to generating prediction
        # what is the label generated?
        y_pred = model(padded_input, seq_lens)  # .cuda()
        loss = model.loss(y_pred, padded_label)  # .cuda()
        # append the loss after converting back to numpy object from tensor
        val_loss.append(loss.data.numpy())
        threshold_output, correct_ones = find_max_predictions(
            y_pred, padded_label, threshold)  # .cuda()
        threshold_output, num_no_pred = mask_padded_errors(
            threshold_output, seq_lens)
        if (random.random() <= perc_sample_print):
            writer_sample_output(output_sample_filename, epoch, step, padded_input,
                                threshold_output, padded_label, correct_ones,
                                exercise_to_index_map, include_correct)
        total_predicted += len(torch.nonzero(threshold_output))
        total_label += len(torch.nonzero(padded_label))
        total_correct += len(torch.nonzero(correct_ones))
        total_no_predicted += num_no_pred
        total_sessions += np.sum(seq_lens)
    average_loss = np.mean(val_loss)
    # of label=1 that were predicted accurately
    return average_loss, total_predicted, total_label, \
        total_correct, total_no_predicted, total_sessions


def mask_padded_errors(threshold_output, seq_lens):
    num_no_pred = 0
    for i, output in enumerate(threshold_output):
        # the full size of threshold
        num_sess = threshold_output[i].shape[0]
        seq_len = seq_lens[i]
        # calculate the number of sessions with no prediction
        sess_with_pred = np.sum(
            threshold_output[i][:seq_len, ].detach().numpy(),
            axis=1)
        num_no_pred += int(np.sum(sess_with_pred == 0))
        threshold_output[i][:seq_len, ]
        for sess_i in range(seq_len, num_sess):
            threshold_output[i][sess_i] = 0
    return threshold_output, num_no_pred


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
    predict_diff = label - threshold_output
    # set_correct_to_one = F.threshold(0.99, 0)
    incorrect_ones = F.threshold(predict_diff, 0.999, 0)
    correct_ones = label - incorrect_ones
    return threshold_output, correct_ones


def find_max_predictions(output, label, threshold):
    '''
        compare the predicted list and the actual rate
        then generate the locaation of correct predictions
        allow for a relative threshold, so that if no
        values above absolute threshold, still return
        selection
    '''
    # find the max prediction for each session
    max_val = torch.max(output, dim=2)[0].detach().numpy()
    # set the relative threshold output to zero
    rel_thresh_output = torch.zeros(output.shape)
    for stud, _ in enumerate(output):
        for sess, _ in enumerate(output[stud]):
            # set the relative threshold to one if within 0.01
            # of max likelihood, threshold greater than 0.05
            rel_thresh =  max_val[stud, sess] - 0.05
            if rel_thresh<0.05:
                rel_thresh = 0.05
            rel_thresh_output[stud, sess] = torch.Tensor((
                output[stud, sess].detach().numpy() >=rel_thresh
                ).astype(float))
    abs_threshold_output = F.threshold(output, threshold, 0)
    threshold_output = torch.max(rel_thresh_output, abs_threshold_output)
    # find the difference between label and prediction
    # where prediction is incorrect (label is one and
    # threshold output 0), then the difference would be 1
    predict_diff = label - threshold_output
    # set_correct_to_one = F.threshold(0.99, 0)
    incorrect_ones = F.threshold(predict_diff, 0.999, 0)
    correct_ones = label - incorrect_ones
    return threshold_output, correct_ones


def writer_sample_output(output_sample_filename, epoch, step, padded_input,
                        threshold_output, padded_label, correct_ones,
                        exercise_to_index_map, include_correct):
    '''
        Randomly sample batches, and students with each batch
        to write data
        [REFORMAT TODO] turn into class and split write student iter
    '''
    index_to_exercise_map = create_index_to_content_map(exercise_to_index_map)
    step_filename = output_sample_filename+'_'+'ep'+str(epoch)+'st'+str(step)
    step_writer = open(step_filename, 'w')
    # iterate over students
    for i, _ in enumerate(padded_label):
        student = 'step'+str(step) + 'batchstud' + str(i)
        stud_input = padded_input[i]
        actual = padded_label[i]
        prediction = threshold_output[i]
        correct = correct_ones[i]
        write_student_sample(step_writer, student, stud_input,
                             actual, prediction, correct,
                             index_to_exercise_map, include_correct)
    step_writer.close()


def write_student_sample(sample_writer, student, stud_input,
                         actual, prediction, correct, index_to_content_map,
                         include_correct):
    '''
        print readable prediciton sample
        for input, output, label expect a matrix that's already
        converted to ones where value above threshold set to 1
    '''
    content_num = len(index_to_content_map)
    for i, label in enumerate(actual):
        # pass over the first one, no prediction made
        if i == 0:
            continue
        if include_correct:
            readable_input = create_readable_list_with_correct(
                stud_input[i], index_to_content_map, content_num)
        else:
            readable_input = create_readable_list(
                stud_input[i], index_to_content_map)
        readable_output = create_readable_list(
            prediction[i], index_to_content_map)
        readable_label = create_readable_list(
            label, index_to_content_map)
        readable_correct = create_readable_list(
            correct[i], index_to_content_map)
        sample_writer.write(student + '\t' +
                            str(readable_input) + '\t' +
                            str(readable_output) + '\t' +
                            str(readable_label) + '\t' +
                            str(readable_correct) + '\n')


def create_readable_list(vect, index_to_content_map):
    '''
       create the readable list of cotent
    '''
    content_list = []
    indices = np.where(vect > 0.01)[0]
    for index in indices:
        content_list.append(index_to_content_map[index+1])
    return content_list


def create_readable_list_with_correct(vect, index_to_content_map, content_num):
    '''
       create the readable list of cotent
    '''
    content_list = []
    indices = np.where(vect[:content_num-1] > 0.01)[0]
    for index in indices:
        content = index_to_content_map[index+1]
        perc_correct = vect[content_num + index].numpy()
        content_list.append((content, str(perc_correct)))
    return content_list


def create_index_to_content_map(content_index):
    '''
        Reverse the content name to index map
    '''
    index_to_content_map = {}
    for content in content_index:
        index = content_index[content]
        index_to_content_map[index] = content
    return index_to_content_map
