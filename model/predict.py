'''
    Read in test data and predict sessions from existing models
'''
import torch
import torch.utils.data as Data
from torch.nn import functional as F
import yaml
import os
import numpy as np
from process_data import convert_token_to_matrix, split_train_and_test_data, extract_content_map
from evaluate import  find_correct_predictions
from torch.autograd import Variable
from gru import GRU_MODEL as gru_model
import pdb


def predict_sessions(model, full_data, keys, content_dim, threshold, output_filename,
              exercise_to_index_map, include_correct):
    '''
        create recommended session for each session
    '''
    data_index = torch.IntTensor(range(len(keys)))
    torch_data_index = Data.TensorDataset(data_index)
    loader = Data.DataLoader(dataset=torch_data_index,
                             batch_size=1,
                             num_workers=2)
    output_writer = open(output_filename, 'w')
    output_writer.write('student' + '\t' +
                    'last_session' + '\t' +
                    'predicted' + '\t' +
                    'actual' + '\t' +
                    'correct' + '\n')

    for step, batch in enumerate(loader):
        # assume we are not batching the data
        # only one student value relevant
        # returns (student_id, num_sess), only first value used
        student = keys[batch[0]][0]
        # grab all the sessions for a student
        sessions = sorted(full_data[student].keys())
        # convert token data to matrix
        input_padded, label_padded, seq_lens = convert_token_to_matrix(
            batch[0].numpy(), full_data, keys, content_dim, include_correct)
        padded_input = Variable(torch.Tensor(
            input_padded), requires_grad=False)  # .cuda()
        padded_label = Variable(torch.Tensor(
            label_padded), requires_grad=False)  # .cuda()
        model.init_hidden()
        y_pred = model(padded_input, seq_lens)  # .cuda()
        threshold_output, correct_ones = find_max_predictions(
            y_pred, padded_label, threshold)
        writer_sample_output(output_writer, student, sessions, padded_input,
                                threshold_output, padded_label, correct_ones,
                                exercise_to_index_map, include_correct)
    output_writer.close()




def find_max_predictions(output, label, threshold):
    '''
        compare the predicted list and the actual rate
        then generate the locaation of correct predictions
    '''
    # find the max prediction for each session
    max_val = torch.max(output, dim=2)[0].detach().numpy()
    # set the relative threshold output to zero
    rel_thresh_output = torch.zeros(output.shape)
    for stud, _ in enumerate(output):
        for sess, _ in enumerate(output[stud]):
            # set the relative threshold to one if within 0.01
            # of max likelihood, threshold greater than 0.05
            rel_thresh =  max_val[stud, sess] - 0.09
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


def writer_sample_output(output_writer, student, sessions, padded_input,
                         threshold_output, padded_label, correct_ones,
                         exercise_to_index_map, include_correct):
    '''
        Randomly sample batches, and students with each batch
        to write data
        [REFORMAT TODO] turn into class and split write student iter
    '''
    index_to_exercise_map = create_index_to_content_map(exercise_to_index_map)
    # iterate over students
    stud_input = padded_input[0]
    actual = padded_label[0]
    prediction = threshold_output[0]
    correct = correct_ones[0]
    write_student_sample(output_writer, student, sessions, stud_input,
                         actual, prediction, correct,
                         index_to_exercise_map, include_correct)


def write_student_sample(sample_writer, student, sessions, stud_input,
                         actual, prediction, correct, index_to_content_map,
                         include_correct):
    '''
        print readable prediciton sample
        for input, output, label expect a matrix that's already
        converted to ones where value above threshold set to 1
    '''
    content_num = len(index_to_content_map)
    for i, label in enumerate(actual):
        student_session = student + '_' + sessions[i]
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
        sample_writer.write(student_session + '\t' +
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



def run_inference():
    print('start')
    loaded_params = yaml.load(open('input/predict_params.yaml', 'r'))
    model_filename = loaded_params['model_filename']
    nb_lstm_units = loaded_params['nb_lstm_units']
    nb_lstm_layers = loaded_params['nb_lstm_layers']
    threshold = loaded_params['threshold']
    batchsize = loaded_params['batchsize']
    include_correct = loaded_params['include_correct']
    exercise_filename = os.path.expanduser(
        loaded_params['exercise_filename'])
    output_filename = os.path.expanduser(
        loaded_params['output_filename'])
    content_index_filename = loaded_params['content_index_filename']
    # creat ethe filename
    file_affix = model_filename
    print(file_affix)
    exercise_to_index_map, content_dim = extract_content_map(
        content_index_filename)
    keys, _,  full_data, = split_train_and_test_data(exercise_filename,
        content_index_filename, test_perc=0)
    # run the gru model
    if include_correct:
        input_dim = content_dim*2
    else:
        input_dim = content_dim

    model = gru_model(input_dim=input_dim,
                      output_dim=content_dim,
                      nb_lstm_layers=nb_lstm_layers,
                      nb_lstm_units=nb_lstm_units,
                      batch_size=batchsize)
    model.load_state_dict(torch.load( model_filename ))
    predict_sessions(model, full_data, keys, content_dim, threshold,
        output_filename, exercise_to_index_map, include_correct)


if __name__ == '__main__':
    run_inference()