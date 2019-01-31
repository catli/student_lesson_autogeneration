import numpy as np
import json
from sklearn.model_selection import train_test_split


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
        student_key = json_keys[student_index]
        num_sess.append(len(json_data[student_key].keys()))
    max_seq = np.max(num_sess)

    # placeholder padded input, padded with additional sessions
    student_padded = np.zeros((batchsize, int(max_seq), content_num), int)
    # padded label
    # label_padded = np.zeros((batchsize, int(max_seq)-1, content_num), int)

    for stud_num, student_index in enumerate(batch_index):
        # the number of columns = possible contents
        student_key = json_keys[student_index]
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
    input_len = np.sum(input_padded, axis=2)
    return input_padded, label_padded, input_len


def split_train_and_test_data(exercise_filename, content_index_filename,
            test_perc):  # t==22, subtraining and validation; t==25, training and testing
    '''
        split the data into training and test by learners
    '''
    exercise_reader = open(exercise_filename, 'r')
    sessions_exercise_json = json.load(exercise_reader)
    train_data = {}
    val_data = {}
    train_ids, val_ids = split_train_and_test_ids(json_data = sessions_exercise_json
                            , test_perc = test_perc)
    for id in train_ids: train_data[id] = sessions_exercise_json[id]
    for id in val_ids: val_data[id] = sessions_exercise_json[id]
    # [TODO] for count_content_num, consider moving this to train.py
    # to expose the json file
    index_reader = open(content_index_filename, 'r')
    exercise_to_index_map = json.load(index_reader)
    content_num = count_content_num(exercise_to_index_map)
    return train_data, val_data, exercise_to_index_map, content_num



def split_train_and_test_ids(json_data, test_perc):
    '''
        split anon ids into test_perc % in test dataset
        and 1-test_perc % in training dataset
    '''
    student_ids = [student for student in json_data]
    train_ids , val_ids = train_test_split(student_ids,
            test_size = 0.2)
    return train_ids, val_ids


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

