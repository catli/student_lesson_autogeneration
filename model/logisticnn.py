"""
    Define a simple logistic model on each session data
    Use a neural network with single layer to mimic the output of
    logistic model

    Use the pytorch modules to create this model
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.tensor as tensor
import torch.nn.functional as F
import numpy as np
import os
import json
import pdb

input_size = 784
num_epochs = 5
batch_size = 100
learning_rate = 0.01

class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # [@zig: is this the right approach to build this model]
        # out = F.sigmoid(self.linear(x))
        out = F.relu(self.linear(x))
        return out



def run_model(model, train_data, content_num):
    '''
        Run model, iterate over batches
    '''
    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate)
    # Training the Model
    for epoch in range(num_epochs):
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate)
        # each batch has only one student
        # [TODO]: create larger batches to run
        for i, student in enumerate(train_data):
            # split student matrix into input and label, which contains
            #    input sessions [0:(number of sessions - 1)]
            #    label sessions [1:number of sessions]
            input_mat, label_mat = segment_input_label_data(train_data, student, content_num)
            # [@zig: model kept returning format error until I convert to float
            # other there easier ways to do this?]
            # https://community.insightdata.com/community/pl/afnqsyit5in1bf35w9ce4frcye
            input_mat = tensor(input_mat).float()
            label_mat = tensor(label_mat).float()
            # Input a matrix that has session_length x column_vectors
            outputs = model(input_mat)
            loss = criterion(outputs, label_mat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i+1) % 100 == 0:
                print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' 
                       % (epoch+1, num_epochs, i+1, len(train_data), loss.data))
    return model


def validate_model(model, test_data, content_index, threshold):
    '''
        Validate the model on test dataset by finding percent
        of correctly guessed content
        Test output 
        # '04yy3C1FC3iLDDLeRCXCUqQb3nLwEZ7/H19e8g4F0/Y='
    '''
    total_correct = 0
    total_predicted = 0
    total_label = 0
    # index_to_content_map = create_index_to_content_map(content_index)
    content_num = len(content_index)
    for student in test_data:
        test_input_mat, test_label_mat = segment_input_label_data(
            test_data, student, content_num)
        test_input_mat = tensor(test_input_mat).float()
        test_label_mat = tensor(test_label_mat).float()
        test_outputs = model(test_input_mat)
        num_correct, num_predicted, num_label = calculate_recall_precision(
            test_outputs, test_label_mat, threshold)
        total_correct+= num_correct
        total_predicted+= num_predicted
        total_label+= num_label
    return total_correct, total_predicted, total_label


def calculate_recall_precision(output, label, threshold):
    '''
        compare the predicted list and the actual rate
        then calculate the recall % and the prediction %
        store
    '''
    # create binary vectors whether a learners should work
    # on an item
    output_ones = set_above_threshold_to_one(output, threshold)
    label_ones = set_above_threshold_to_one(label, 1)
    correct_prediction = output_ones * label_ones
    num_correct = np.sum(np.array(correct_prediction.detach()))
    num_predicted = np.sum(np.array(output_ones.detach()))
    num_label = np.sum(np.array(label_ones.detach()))
    return num_correct, num_predicted, num_label



def set_above_threshold_to_one(output, threshold):
    '''
        set values above treshold to one and otherwise
        as zero
    '''
    output[output>=threshold]=1
    output[output<threshold]=0
    return output













###################
## return sample
def return_output_above_threshold_list(output, treshold):
    '''
        set the output threshold at which you want
        to recommend the content
    '''
    output_above_threshold = []
    for i, val in enumerate(output):
        if val>=threshold:
            # offset by one to match content index
            output_above_threshold.append(i+1)
    return output_above_threshold

def create_index_to_content_map(content_index):
    '''
        Reverse the content name to index map
    '''
    index_to_content_map = {}
    for content in content_index:
        index = content_index[content]
        index_to_content_map[index] = content_name
    return index_to_content_map



def map_index_to_name(predicted_list, index_to_content_map):
    '''
        map the list of items back to the name
    '''
    exercise_name = []
    for item in predicted_list:
        exercise_name.append(
            index_to_content_map[item])
    return exercise_name


#############################################

def split_test_and_train_data(data):
    '''
        split data
        #T[TODO] split the loaded dataset into test and training
    '''



def segment_input_label_data(data, student, content_num):
    '''
        For the specified student generate an input
        and output matrix from the training data
        currently the logic is simply taking the entire
        student matrix with number of rows = number of sessions
        and partitioning:
            input sessions [0:(number of sessions - 1)]
            label sessions [1:number of sessions]
    '''
    student_json = data[student]
    # student_matrix: number of sessions x possible content type
    student_matrix = convert_token_to_matrix(student_json, content_num)
    # split student matrix into input and label, which contains
    #    input sessions [0:(number of sessions - 1)]
    #    label sessions [1:number of sessions]
    input_mat, label_mat = split_input_label(student_matrix)
    return input_mat, label_mat



def convert_token_to_matrix(student_json, content_num):
    '''
        convert the token to a one-hot vector
        from self.exercise_data
    '''
    sessions = sorted(student_json.keys())
    student_matrix = np.zeros((len(sessions), content_num))
    for sess_num, session in enumerate(sessions):
        # the number of columns = possible contents
        sess_vect = np.zeros((content_num))
        content_items = student_json[session]
        for item_num, item in enumerate(content_items):
            exercise_id = item[0]
            # [TODO] add is_correct as a dimension
            is_correct = item[1]
            # replace the ith content sequence
            # with the identified exercise_id
            # the index of the item starts 0
            sess_vect[exercise_id-1] += 1.0
        # [TODO] For sigmoid prediction
        #    normalize the prediction output between
        #         0 to 1, dividing by total content
        #
        # for empty content, with no exercise
        # set first column as 1
        student_matrix[sess_num, :] = sess_vect
    return student_matrix


def split_input_label(student_matrix):
    '''
        split the matrix of student data into input and labels
        all sessions up to the last one will be used to predict
        the future sessions
    '''
    # all matrices from 1st session to second to last one
    input_mat = student_matrix[0:-1,:]
    # all matrices from 2nd session to last one
    output_mat = student_matrix[1:,:]
    return input_mat, output_mat



exercise_filename = os.path.expanduser(
            '~/sorted_data/khan_problem_json_small')
content_index_filename = 'data/exercise_index'

exercise_reader = open(exercise_filename, 'r')
index_reader = open(content_index_filename, 'r')
sessions_exercise_json = json.load(exercise_reader)


exercise_to_index_map = json.load(index_reader)
content_num = len(exercise_to_index_map)
model = LogisticRegression(input_size = content_num,
            output_size= content_num)
# load an run the model
# [TODO] split dataset into train / test
model = run_model(model, sessions_exercise_json, content_num)
corrects, predictions, labels = validate_model(
    model, sessions_exercise_json, exercise_to_index_map, 0.3)
print('%d / %d precision and %d / %d recall' % (
    corrects, predictions, corrects, labels))




