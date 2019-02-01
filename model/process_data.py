import numpy as np
import json
from sklearn.model_selection import train_test_split
import torch.nn.utils as utils
import pdb


def convert_token_to_matrix(batch_index, json_data, json_keys, content_num):
    '''
        convert the token to a multi-hot vector
        from student session activity json_data
        convert this data in batch form
    '''
    # number of students in the batch
    batchsize = len(batch_index)
    num_sess = []
    # max number of sessions in batch
    for student_index in batch_index:
        # return the key pairs (student_id, seq_len)
        # and the first item of pair as student id
        student_key = json_keys[student_index][0]
        num_sess.append(len(json_data[student_key].keys())-1)
    max_seq = np.max(num_sess) + 1

    # placeholder padded input, padded with additional sessions
    student_padded = np.zeros((batchsize, int(max_seq), content_num), int)

    # populate student_padded
    for stud_num, student_index in enumerate(batch_index):
        # return the key pairs (student_id, seq_len)
        # and the first item of pair as student id
        student_key = json_keys[student_index][0]
        sessions = sorted(json_data[student_key].keys())
        for sess_num, session in enumerate(sessions):
            content_items = json_data[student_key][session]
            for item_num, item in enumerate(content_items):
                exercise_id = item[0]
                is_correct = item[1]
                # [TODO] add correct
                student_padded[stud_num, sess_num, exercise_id-1]=1
    input_padded = student_padded[:, :-1]
    label_padded = student_padded[:, 1:]
    # assign the number of sessions as sequence length for each student
    # this will feed be used later to tell the model
    # which sessions are padded
    seq_lens = num_sess
    return input_padded, label_padded, seq_lens


def split_train_and_test_data(exercise_filename, content_index_filename,
            test_perc):
    '''
        split the data into training and test by learners
    '''
    exercise_reader = open(exercise_filename, 'r')
    full_data = json.load(exercise_reader)
    train_data = {}
    val_data = {}
    ordered_train_keys, ordered_val_keys = split_train_and_test_ids(
                        json_data = full_data,
                        test_perc = test_perc)
    # for id in train_ids: train_data[id] = sessions_exercise_json[id]
    # for id in val_ids: val_data[id] = sessions_exercise_json[id]
    # [TODO] for count_content_num, consider moving this to train.py
    # to expose the json file
    index_reader = open(content_index_filename, 'r')
    exercise_to_index_map = json.load(index_reader)
    content_num = count_content_num(exercise_to_index_map)
    return ordered_train_keys, ordered_val_keys, full_data, content_num



def split_train_and_test_ids(json_data, test_perc):
    '''
        split anon ids into test_perc % in test dataset
        and 1-test_perc % in training dataset
    '''
    student_ids = [student for student in json_data]
    train_ids , val_ids = train_test_split(student_ids,
            test_size = 0.2)
    ordered_train_keys = create_ordered_sequence_list(train_ids, json_data)
    ordered_val_keys = create_ordered_sequence_list(val_ids, json_data)
    return ordered_train_keys, ordered_val_keys


def create_ordered_sequence_list(set_ids, exercise_json):
    '''
        create ordered sequence length
        will be used for batching to efficiently
        run through the sequence ids
    '''
    key_seq_pair = create_key_seqlen_pair(set_ids, exercise_json)
    key_seq_pair.sort(key = lambda x: x[1], reverse = True)
    return key_seq_pair


def create_key_seqlen_pair(set_ids, json_data):
    '''
        create a tuple with learner id and the sequence length,
        i.e. number of sessions per learner
            ('$fd@w', 4)
    '''
    key_seq_pair  = []
    for id in set_ids:
        seq_len = len(json_data[id])
        key_seq_pair.append((id, seq_len))
    return key_seq_pair


def count_content_num(exercise_map):
    return len(exercise_map.keys())


# def split_input_label(student_matrix, input_lag):
#     '''
#         split the matrix of student data into input and labels
#         all sessions up to the last one will be used to predict
#         the future sessions
#     '''
#     # all matrices from 1st session to second to last one
#     input_mat = student_matrix[0:-1,:].copy()
#     # all matrices from 2nd session to last one
#     output_mat = student_matrix[1:,:].copy()
#     input_mat_lag = input_mat.copy()
#     # create a lagged average input of multiple sessions
#     if input_lag == 1:
#         lag_input = np.vstack((
#                         np.zeros(student_matrix.shape[1]),
#                         student_matrix[0:-2,:].copy()))
#         input_mat_lag = input_mat + lag_input
#     if input_lag == 2:
#         lag_input_1 = np.vstack((
#                         np.zeros(student_matrix.shape[1]),
#                         student_matrix[0:-2,:].copy()))
#         lag_input_2 = np.vstack((
#                         np.zeros((2,student_matrix.shape[1])),
#                         student_matrix[0:-3,:].copy()))
#         input_mat_lag = input_mat + lag_input_1 + lag_input_2
#     return input_mat_lag, output_mat, input_mat

