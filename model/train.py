'''
    Train the model
    for batched data, batch and train
    the approach to training takes a lot of inspiration from how this
    CAHL goal-based model was built https://github.com/CAHLR/goal-based-recommendation
    by WeiJiang (@fabulosa)
'''

from GRU import GRU as gru_model
import torch
import torch.nn as nn
from logisticnn import Neural_Network
from process_data import split_train_and_test_data, convert_token_to_matrix
from evaluate import evaluate_loss, evaluate_precision_and_recall
from sklearn.model_selection import train_test_split
import pdb



def train_and_evaluate(model, train_data, val_data, optimizer):
    best_vali_loss = None  # set a large number for validation loss at first
    best_vali_accu = 0
    epoch = 0
    training_loss_epoch = []
    testing_loss_epoch = []
    #[TODO] decide how to feed in content_num variable
    content_num = #PLACEHOLDER
    epoch = 10 # PLACEHOLDER
    # training data on mini batch
    # [TODO] how to save the training data
    train_data_index = torch.IntTensor(range(train_data.shape[0]))
    torch_train_data_index = Data.TensorDataset(data_tensor=train_data_index, target_tensor=train_data_index)
    train_loader = Data.DataLoader(dataset=torch_train_data_index, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)
    # validation data on mini batch
    val_data_index = torch.IntTensor(range(val_data.shape[0]))
    torch_val_data_index = Data.TensorDataset(data_tensor=val_data_index, target_tensor=val_data_index)
    vali_loader = Data.DataLoader(dataset=torch_val_data_index, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)
    while True:
        epoch += 1
        train_loss = train(model, optimizer, train_loader, train_data, epoch, content_dim)
        training_loss_epoch.append(train_loss)
        print('The average loss of training set for the first ' + str(epoch) + ' epochs: ' + str(training_loss_epoch))
        eval_loss = evaluate_loss(model, loader, val_data, content_dim)
        testing_loss_epoch.append(eval_loss)
        num_predicted, num_label, num_correct = evaluate_precision_and_recall(model, vali_loader, val_data, batchsize, dim_input_course, dim_input_grade, dim_input_major, weight3, weight4)
        print('Epoch test: %d / %d = %f precision and %d / %d = %f recall' % (
                num_correct, num_predicted, num_correct/num_predicted,
                num_correct, num_label, num_correct/num_label))
        if epoch >= 5:
            # [TODO] consider adding an early stopping logic
            break


def train(model, optimizer, loader, train_data, epoch, content_dim):
    # set in training node
    model.train()
    train_loss = []
    for step, (batch_x, batch_y) in enumerate(loader):  # batch_x: index of batch data
        print('Epoch: ', epoch, ' | Iteration: ', step+1)
        processed_data = convert_token_to_matrix(batch_x.numpy(), train_data, content_dim)
        # Variable, used to set tensor, but no longer necessary
        # Autograd automatically supports tensor with requires_grade=True
        #  https://pytorch.org/docs/stable/autograd.html?highlight=autograd%20variable
        seq_len = processed_data[1]
        padded_input = tensor(torch.Tensor(processed_data[0]), requires_grad=False).cuda()
        padded_label = tensor(torch.Tensor(processed_data[2]), requires_grad=False).cuda()

        # clear gradients and hidden state
        optimizer.zero_grad()
        model.hidden = model.init_hidden()
        model.hidden[0] = model.hidden[0].cuda()
        model.hidden[1] = model.hidden[1].cuda()
        # is this equivalent to generating prediction
        # what is the label generated?
        y_pred = model(padded_input, seq_len).cuda()
        loss = model.loss(y_pred, padded_label).cuda()
        print('Epoch ' + str(epoch) + ': ' + 'The '+str(step+1)+'-th interation: loss'+str(loss.data[0])+'\n')
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data[0])

    average_loss = np.mean(train_loss)
    return average_loss


if __name__ == '__main__':
    # only consider grade higher than B or not, pass or not pass

    # set hyper parameters
    nb_lstm_units = 1000
    nb_lstm_layers = 1
    batchsize = 20
    learning_rate = 0.001
    test_perc = 0.2
    exercise_filename = os.path.expanduser(
                '~/sorted_data/khan_problem_token_3only')
    content_index_filename = 'data/exercise_index_3only'
    train_data, val_data = split_train_and_test_data(exercise_filename, content_index_file, test_perc)
    model = gru_model(input_dim, output_dim, nb_lstm_layers, nb_lstm_units, batch_size)
    # [TODO] consider whether to include weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_and_evaluate(model, train_data, val_data, optimizer)
