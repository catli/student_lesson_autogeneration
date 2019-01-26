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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
import pdb


class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        #[ @zig: is this the right way to build out the logistic
        #    model in pytorch ]
        out = F.sigmoid(self.linear(x))
        # out = F.relu(self.linear(x))
        return out



def run_model(model, train_data, content_index, learning_rate, num_epochs):
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
    loss_trend = []
    for epoch in range(num_epochs):
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate)
        # store the array of loss trend to visualize
        # each batch has only one student
        # [TODO]: create larger batches to run
        # [ @zig: I haven't done a good job batching the dataset.
        #     I'm running the gradient step for every student matrix
        #     as a separate batch and I want to batch it by stacking
        #     n number of student input and label matrices together
        #    is there a better way of doing this? ]
        total_epoch_loss = 0
        for i, student in enumerate(train_data):
            # split student matrix into input and label, which contains
            #    input sessions [0:(number of sessions - 1)]
            #    label sessions [1:number of sessions]
            input_mat, label_mat = segment_input_label_data(train_data, student, content_index)
            # normalize dataset for logistic model
            # setting output greater than 1 to 1 ehre
            # [@zig: I'm setting the label to one here since I figure
            #    logistic regression and sigmoid function would not work well
            #    otherwise, is this the right way to do it?]
            input_mat, label_mat = normalize_or_threshold(input_mat, label_mat)

            #[ @zig: model kept returning format error until I convert to float
            # other there easier ways to do this?]
            # https://community.insightdata.com/community/pl/afnqsyit5in1bf35w9ce4frcye
            input_mat = tensor(input_mat).float()
            label_mat = tensor(label_mat).float()

            # Input a matrix that has session_length x column_vectors
            outputs = model(input_mat)
            loss = criterion(outputs, label_mat)
            # add the epoc_
            total_epoch_loss+=loss.data.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i+1) % 100 == 0:
                print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' 
                       % (epoch+1, num_epochs, i+1, len(train_data), loss.data))
        # [TODO] update once batch logic put in place
        loss_trend.append(total_epoch_loss)
        print(total_epoch_loss)
    print('# of students: %d' % i)
    return model, loss_trend



def validate_model(model, test_data, content_index, threshold):
    '''
        Validate the model on test dataset by finding percent
        of correctly guessed content
        Test output 
        # small: '04yy3C1FC3iLDDLeRCXCUqQb3nLwEZ7/H19e8g4F0/Y='
        # tiny: '000JXAzy89wFPdoDMC1ySdZ/fbrGT6/p3djoBshJI0g='
        # tiny test: '000j+MxoGUQ7M9LFL2reKkbGlwR8+iefk/am1iQN/fo=''
    '''
    # create a mapping from content back to index
    # so it can be visualized
    index_to_content_map = create_index_to_content_map(content_index)
    total_correct = 0
    total_predicted = 0
    total_label = 0
    for student in test_data:
        test_input_mat, test_label_mat = segment_input_label_data(
            test_data, student, content_index)
        test_input_mat, test_label_mat = normalize_or_threshold(
            test_input_mat, test_label_mat)
        test_input_mat = tensor(test_input_mat).float()
        test_label_mat = tensor(test_label_mat).float()
        test_outputs = model(test_input_mat)
        num_correct, num_predicted, num_label = calculate_recall_precision(
            test_outputs, test_label_mat, threshold, index_to_content_map)
        total_correct+= num_correct
        total_predicted+= num_predicted
        total_label+= num_label
    return total_correct, total_predicted, total_label





def segment_input_label_data(data, student, content_index):
    '''
        For the specified student generate an input
        and output matrix from the training data
        currently the logic is simply taking the entire
        student matrix with number of rows = number of sessions
        and partitioning:
            input sessions [0:(number of sessions - 1)]
            label sessions [1:number of sessions]
    '''
    content_num = len(content_index)
    json_data = data[student]
    # student_matrix: number of sessions x possible content type
    student_matrix = convert_token_to_matrix(json_data, content_num)
    # split student matrix into input and label, which contains
    #    input sessions [0:(number of sessions - 1)]
    #    label sessions [1:number of sessions]
    input_mat, label_mat = split_input_label(student_matrix)
    return input_mat, label_mat



def convert_token_to_matrix(json_data, content_num):
    '''
        convert the token to a multi-hot vector
        from student session activity json_data
    '''
    sessions = sorted(json_data.keys())
    student_matrix = np.zeros((len(sessions), content_num))
    for sess_num, session in enumerate(sessions):
        # the number of columns = possible contents
        sess_vect = np.zeros((content_num))
        content_items = json_data[session]
        for item_num, item in enumerate(content_items):
            exercise_id = item[0]
            # [TODO] add is_correct as a dimension
            is_correct = item[1]
            # add the count for the session vector
            # with the identified exercise_id
            # the index of the item starts 0
            sess_vect[exercise_id-1] += 1
        student_matrix[sess_num, :] = sess_vect
    return student_matrix


# [TODO] add or average input matrices from multiple timestamps
# def convert_token_to_average_matrix(json_data, content_num):
#     '''
#         convert the token to a one-hot vector
#         from student session activity json_data
#     '''
#     sessions = sorted(json_data.keys())
#     student_matrix = np.zeros((len(sessions), content_num))
#     for sess_num, session in enumerate(sessions):
#         # the number of columns = possible contents
#         sess_vect = np.zeros((content_num))
#         content_items = json_data[session]
#         for item_num, item in enumerate(content_items):
#             exercise_id = item[0]
#             # [TODO] add is_correct as a dimension
#             is_correct = item[1]
#             # add the count for the session vector
#             # with the identified exercise_id
#             # the index of the item starts 0
#             sess_vect[exercise_id-1] = 1.0
#         # [TODO] add the last sess
#         # sess_vect
#         student_matrix[sess_num, :] = sess_vect
#         last_sess = sess_vect
#     return student_matrix




def split_input_label(student_matrix):
    '''
        split the matrix of student data into input and labels
        all sessions up to the last one will be used to predict
        the future sessions
    '''
    # all matrices from 1st session to second to last one
    input_mat = student_matrix[0:-1,:].copy()
    # all matrices from 2nd session to last one
    output_mat = student_matrix[1:,:].copy()
    return input_mat, output_mat



def normalize_or_threshold(input_mat, output_mat):
    '''
        adjust the input and output matrices as necessary
    '''
    # input_mat = normalize_vector(input_mat)
    output_mat = set_above_threshold_to_one_np(output_mat, 1)
    # output_mat = normalize_vector(output_mat)
    return input_mat, output_mat



def normalize_vector(matrix):
    '''
        for each vector in a matrix normalize by dividing by the total
        normalize the prediction output between
        0 to 1, dividing by total content
    '''
    for row, vect in enumerate(matrix):
        matrix[row] = vect/np.sum(vect)
    return matrix


def set_above_threshold_to_one_np(matrix, threshold):
    '''
        for each vector in a matrix normalize by dividing by the total
        normalize the prediction output between
        0 to 1, dividing by total content
    '''
    matrix[matrix>=threshold] = 1
    matrix[matrix<threshold] = 0
    return matrix


def set_above_threshold_to_one(output, threshold):
    '''
        set values above treshold to one and otherwise
        as zero
    '''
    return torch.where(output>=threshold,
        torch.ones(output.shape), torch.zeros(output.shape))


def calculate_recall_precision(output, label, threshold, index_to_content_map):
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
    # if np.sum(correct_prediction.numpy())>=1:
    #     print_prediction_sample(output_ones,
    #         label_ones, correct_prediction, index_to_content_map)
    num_correct = np.sum(np.array(correct_prediction.detach()))
    num_predicted = np.sum(np.array(output_ones.detach()))
    num_label = np.sum(np.array(label_ones.detach()))
    return num_correct, num_predicted, num_label




def split_train_and_test_data(data, test_perc):
    '''
        split anon ids into test_perc % in test dataset
        and 1-test_perc % in training dataset
    '''
    train_data = {}
    test_data = {}
    train_ids, test_ids = split_train_and_test_ids(data, test_perc)
    for id in train_ids: train_data[id] = data[id]
    for id in test_ids: test_data[id] = data[id]
    return train_data, test_data


def split_train_and_test_ids(data, test_perc):
    '''
        split anon ids into test_perc % in test dataset
        and 1-test_perc % in training dataset
    '''
    student_ids = [student for student in data]
    train_ids , test_ids = train_test_split(student_ids,
            test_size = 0.2)
    return train_ids, test_ids


def print_prediction_sample(output, label, correct, index_to_content_map):
    '''
        print readable prediciton sample
    '''
    readable_ouput = create_readable_list(output, index_to_content_map)
    readable_label = create_readable_list(label, index_to_content_map)
    readable_correct = create_readable_list(correct, index_to_content_map)
    print('PREDICTED OUTPUT')
    print(readable_ouput)
    print('LABEL')
    print(readable_label)
    print('CORRECT')
    print(readable_correct)


def create_readable_list(matrix, index_to_content_map):
    '''
       create the readable list of contents
    '''
    content = []
    for vect in matrix:
        indices = np.where(vect.numpy() >= 1)[0]
        for index in indices:
            content.append(index_to_content_map[index+1])
    return content


def print_out_readable_student_data(json_data, content_index, filename):
    index_to_content_map = create_index_to_content_map(content_index)
    readable_data = {}
    for student in json_data:
        student_activities = []
        sessions = sorted(json_data[student].keys())
        readable_data[student] = {}
        for session in sessions:
            session_activities =[]
            session_indices= json_data[student][session]
            for ex in session_indices:
                session_activities.append(
                    index_to_content_map[ex[0]])
            readable_data[student][session] = session_activities
    writer = open(filename, 'w')
    json.dump(readable_data, writer)


def create_index_to_content_map(content_index):
    '''
        Reverse the content name to index map
    '''
    index_to_content_map = {}
    for content in content_index:
        index = content_index[content]
        index_to_content_map[index] = content
    return index_to_content_map





def main():
    '''
        Read files locally, open, and then run model
        Generate the prediction performance
    '''
    # set parameters
    num_epochs = 600
    learning_rate = 1
    # load data
    exercise_filename = os.path.expanduser(
                '~/sorted_data/khan_problem_json_small')
    content_index_filename = 'data/exercise_index'

    exercise_reader = open(exercise_filename, 'r')
    index_reader = open(content_index_filename, 'r')
    sessions_exercise_json = json.load(exercise_reader)
    # split data to train and test
    train_data, test_data = split_train_and_test_data(sessions_exercise_json,
                                test_perc= 0.2)
    exercise_to_index_map = json.load(index_reader)
    content_num = len(exercise_to_index_map)
    # instantiate the linear model
    model = LogisticRegression(input_size = content_num,
                output_size= content_num)
    # run the model
    model, loss_trend = run_model(model, train_data,
                exercise_to_index_map, learning_rate, num_epochs)
     # visualize the loss
    plt.plot(range(len(loss_trend)), loss_trend, 'r--')
    plt.savefig(os.path.expanduser('~/Downloads/loss_function.jpg'))
    # validate the model
    corrects, predictions, labels = validate_model(
        model, test_data, exercise_to_index_map, 0.5)
    print('%d / %d = %f precision and %d / %d = %f recall' % (
        corrects, predictions, corrects/predictions,
        corrects, labels, corrects/labels))

    print('OUTPUT READABLE TRAIN DATA')
    print_out_readable_student_data(train_data, exercise_to_index_map,
        os.path.expanduser('~/Downloads/readable_train'))
    print('OUTPUT READABLE TEST DATA')
    print_out_readable_student_data(test_data, exercise_to_index_map,
        os.path.expanduser('~/Downloads/readable_test'))


if __name__ == '__main__':
    main()








# ###################
# ## return sample
# def return_output_above_threshold_list(output, treshold):
#     '''
#         set the output threshold at which you want
#         to recommend the content
#     '''
#     output_above_threshold = []
#     for i, val in enumerate(output):
#         if val>=threshold:
#             # offset by one to match content index
#             output_above_threshold.append(i+1)
#     return output_above_threshold

# def map_index_to_name(predicted_list, index_to_content_map):
#     '''
#         map the list of items back to the name
#     '''
#     exercise_name = []
#     for item in predicted_list:
#         exercise_name.append(
#             index_to_content_map[item])
#     return exercise_name


# #############################################
