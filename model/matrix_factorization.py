'''
    Try out matrix factorization on the dataset and
    check how well it predicts 
    https://www.ethanrosenthal.com/2017/06/20/matrix-factorization-in-pytorch/    

'''

import numpy as np
from scipy.sparse import rand as sprand
import torch
from torch.autograd import Variable

# Make up some random explicit feedback ratings
# and convert to a numpy array
# n_users = 1000
# n_items = 1000
# ratings = sprand(n_users, n_items, 
#                  density=0.01, format='csr')
# ratings.data = (np.random.randint(1, 5, 
#                                   size=ratings.nnz)
#                           .astype(np.float64))
# ratings = ratings.toarray()

class MatrixFactorization(torch.nn.Module):

    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, 
                                               n_factors,
                                               sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, 
                                               n_factors,
                                               sparse=True)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item))


def run_model(model, train_data, content_index):
    '''
        Run model, iterate over matrix fthe dataset
    '''
    loss_func = torch.nn.MSELoss()
    model = MatrixFactorization(n_users, n_items, n_factors=20)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=1e-6) # learning rate

    batch_ids = create_training_batch(train_data, batch_size)
    # Sort our data
    rows, cols = ratings.nonzero()
    p = np.random.permutation(len(rows))
    rows, cols = rows[p], cols[p]

    for i, student_key in enumerate(train_data):

        # convert token data to matrix
        # need to convert batch_x from tensor flow object to numpy array
        # before converting to matrix
        input_mat_lag, label_mat, __ = segment_input_label_data(
            train_data, student_key, content_index,)

        # Turn data into variables
        rating = Variable(torch.FloatTensor([ratings[row, col]]))
        row = Variable(torch.LongTensor([np.long(row)]))
        col = Variable(torch.LongTensor([np.long(col)]))

        # Predict and calculate loss
        prediction = model(row, col)
        loss = loss_func(prediction, rating)

        # Backpropagate
        loss.backward()
        # Update the parameters
        optimizer.step()
    return model

def evaluate_model():
    '''
        [TODO] How to evaluate the model
    '''



# Segment and data into batches for learning
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
    student_data = data[student]
    # student_matrix: number of sessions x possible content type
    # split student matrix into input and label, which contains
    #    input sessions [0:(number of sessions - 1)]
    #    label sessions [1:number of sessions]
    label_mat, input_mat = convert_token_to_matrix(
        student_data, content_num)
    return label_mat, input_mat



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
            sess_vect[exercise_id-1] = 1 + is_correct*2
        student_matrix[sess_num, :] = sess_vect
    # split student matrix into input and label, which contains
    #    input sessions [0:(number of sessions - 1)]
    #    label sessions [1:number of sessions]
    label_mat,  input_mat = split_input_label(student_matrix)
    return label_mat, input_mat



def split_input_label(student_matrix):
    '''
        split the matrix of student data into input and labels
        all sessions up to the last one will be used to predict
        the future sessions
    '''
    # all matrices from 1st session to second to last one
    input_mat = student_matrix[0:-1, :].copy()
    # all matrices from 2nd session to last one
    output_mat = student_matrix[1:, :].copy()
    return output_mat, input_mat




