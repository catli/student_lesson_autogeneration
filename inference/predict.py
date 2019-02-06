from model.process_data import  convert_token_to_matrix, split_train_and_test_data
from model.evaluate import  find_correct_predictions

'''
    Read in test data and predict sessions from existing models
'''
def predict_sessions(model, full_data, keys, content_dim, threshold, output_filename,
              exercise_to_index_map, include_correct):
    model.eval()
    data_index = torch.IntTensor(range(len(keys)))
    torch_data_index = Data.TensorDataset(data_index)
    loader = Data.DataLoader(dataset=torch_data_index,
                             batch_size=1,
                             num_workers=2)
    output_writer = open(output_filename, 'w')
    for step, batch in enumerate(loader):
        # convert token data to matrix
        student = keys[batch_index]
        sessions = full_data[student].keys
        input_padded, label_padded, seq_lens = convert_token_to_matrix(
            batch[0].numpy(), full_data, keys, content_dim, include_correct)
        padded_input = Variable(torch.Tensor(
            input_padded), requires_grad=False)  # .cuda()
        padded_label = Variable(torch.Tensor(
            label_padded), requires_grad=False)  # .cuda()
        model.hidden = model.init_hidden()
        y_pred = model(padded_input, seq_lens)  # .cuda()
        threshold_output, correct_ones = find_correct_predictions(
            y_pred, padded_label, threshold)
        writer_sample_output(output_writer, student, sessions, padded_input,
                                threshold_output, padded_label, correct_ones,
                                exercise_to_index_map, include_correct)


def writer_sample_output(output_writer, student, sessions, padded_input,
                         threshold_output, padded_label, correct_ones,
                         exercise_to_index_map, include_correct):
    '''
        Randomly sample batches, and students with each batch
        to write data
        [REFORMAT TODO] turn into class and split write student iter
    '''
    index_to_exercise_map = create_index_to_content_map(exercise_to_index_map)
    step_filename = output_sample_filename
    # iterate over students
    for i, _ in enumerate(padded_label):
        student_session = student + '_' + sessions[i]
        stud_input = padded_input[i]
        actual = padded_label[i]
        prediction = threshold_output[i]
        correct = correct_ones[i]
        write_student_sample(output_writer, student_session, stud_input,
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



def run_inference():
    loaded_params = yaml.load(open('model_params.yaml', 'r'))
    model_filename = loaded_params['model_filename']
    threshold = loaded_params['threshold']
    batchsize = loaded_params['batchsize']
    include_correct = loaded_params['include_correct']
    exercise_filename = os.path.expanduser(
        loaded_params['exercise_filename'])
    output_filename = os.path.expanduser(
        loaded_params['output_sample_filename'])
    content_index_filename = loaded_params['content_index_filename']
    # creat ethe filename
    file_affix = model_filename
    model = torch.load( model_filename )
    exercise_to_index_map, content_dim = extract_content_map(
        content_index_filename)
    keys, full_data = split_train_and_test_data(exercise_filename,
        content_index_filename, test_perc=0)
    predict_sessions(model, full_data, keys, content_dim, threshold,
        output_filename, exercise_to_index_map, include_correct)


if __name__ = "main":
    run_inference()