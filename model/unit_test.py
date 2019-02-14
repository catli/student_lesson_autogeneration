import torch
import torch.nn as nn
import torch.utils.data as Data
from gru import GRU_MODEL as gru_model
from process_data import split_train_and_test_data, convert_token_to_matrix, extract_content_map
from train import train
import pdb


def test_train():
    exercise_filename = 'data/fake_tokens'
    content_index_filename = 'data/exercise_index_all'
    train_keys, val_keys, full_data = split_train_and_test_data(
        exercise_filename, content_index_filename, 0)
    exercise_to_index_map, content_dim = extract_content_map(
        content_index_filename)
    input_dim = content_dim*2
    model = gru_model(input_dim=input_dim,
                      output_dim=content_dim,
                      nb_lstm_layers=1,
                      nb_lstm_units=50,
                      batch_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data_index = torch.IntTensor(range(len(train_keys)))
    torch_data_index = Data.TensorDataset(data_index)
    loader = Data.DataLoader(dataset=torch_data_index,
                                       batch_size=1,
                                       drop_last=True)
    train(model, optimizer, full_data, loader, train_keys, epoch = 1, 
          content_dim = content_dim, include_correct = True)
    assert model, "UH OH"
    print("PASS UNIT TEST")



if __name__ == '__main__':
    # set hyper parameters
    test_train()
