"""
    Process data
    Transform tabular data into
    concatenated one-hot vector with
    each vector representing a student session
    format of matrix
        (student batch, sessions, input dimensions)
"""
import json
import time
import os
import numpy as np
import pdb
import pickle


class ProcessData():

    def __init__(self, exercise_filename = '', exercise_index_file = ''):
        print('exercise file %s' % exercise_filename)
        self.exercise_reader = open(exercise_filename,'r')
        self.exercise_index_reader = open(exercise_index_file,'r')
        self.exercise_index = json.load(self.exercise_index_reader)
        self.exercise_data = {}

    def dump_exercise_file_into_json(self, write_filename):
        '''
            iterate through exercise line
            and produce a json file with the
            student id as the key 
            # [TODO] should this product json or numpy array?
        '''
        first_line = self.exercise_reader.readline().strip()
        col_names = first_line.split(',')
        student_id_loc = col_names.index('sha_id')
        session_id_loc = col_names.index('session_start_time')
        exercise_loc = col_names.index('exercise')
        correct_loc = col_names.index('correct')
        counter = 0
        for line in self.exercise_reader:
            line_delimited = line.strip().split(',')
            student_id = line_delimited[student_id_loc]
            session_id = line_delimited[session_id_loc]
            exercise = line_delimited[exercise_loc]
            is_correct = line_delimited[correct_loc]
            self.add_new_exercise_data(student_id,
                session_id, exercise, is_correct)
            counter+=1
            if counter % 1000000 == 0:
                print(counter)
        self.delete_single_session_user()
        self.write_json_file(write_filename)
        return self.exercise_data

    def delete_single_session_user(self):
        '''
            delete any learner with a single session
            will not be useful for training
        '''
        students_to_delete = []
        for student in self.exercise_data:
            sessions = len(self.exercise_data[student])
            if sessions == 1:
                students_to_delete.append(student)
        for student in students_to_delete:
            del self.exercise_data[student]



    def add_new_exercise_data(self, student_id, session_id, exercise, is_correct):
        # append exercise data to output
        if student_id not in self.exercise_data:
            self.exercise_data[student_id] = {}
            self.exercise_data[student_id][session_id] = []
        elif session_id not in self.exercise_data[student_id]:
            self.exercise_data[student_id][session_id] = []
        exercise_id = self.exercise_index['exercise:'+exercise]
        correct = int(is_correct == 'true')
        self.exercise_data[student_id][session_id].append((exercise_id, correct))


    def write_json_file(self, write_filename):
        # store the json file into output
        json_writer = open(write_filename, 'w')
        print('writer file with %f students:' % len(self.exercise_data))
        json.dump(self.exercise_data, json_writer)


    def convert_token_to_matrix(self, json_data, write_filename):
        '''
            convert the token to a one-hot vector
            from self.exercise_data
        '''
        # an array that maintains the order the student batches
        # are produced in, so it can be used to generate training
        # vs test
        self.student_index = []
        self.student_matrices = {}
        content_num = len(self.exercise_index)
        for student in json_data:
            self.student_index.append(student)
            student_json = json_data[student]
            sessions = sorted(student_json.keys())
            student_matrix = np.zeros(
                    (len(sessions), 200*(content_num+1)))
            for sess_num, session in enumerate(sessions):
                # the number of columns = possible contents
                # plus one additional content when non content
                sess_mat = np.zeros((200, content_num+1))
                content_items = student_json[session]
                # truncate content items to first 200
                # [TODO]: consider extending or varying read
                #        sequence length
                content_items = content_items[:200]
                for item_num,item in enumerate(content_items):
                    # item contains (exercise_id, is_correct)
                    exercise_id = item[0]
                    # [TODO] add is_correct as a dimension
                    is_correct = item[1]
                    # replace the ith content sequence
                    # with the identified exercise_id
                    # the index of the item starts at the beginning 
                    sess_mat[item_num, exercise_id] = 1
                # for empty content, with no exercise
                # set first column as 1
                sess_mat[len(content_items):200, 0] = 1
                sess_vect = sess_mat.reshape(200*(content_num+1),)
                student_matrix[sess_num, :] = sess_vect
            #self.student_matrices[student] = student_matrix
        # print(self.student_matrices)
        # matrix_writer = open(write_filename, 'w')
        np.save(write_filename, student_matrix)
        pdb.set_trace()




if __name__ == '__main__':
    start = time.time() 
    exercise_filename = os.path.expanduser(
            '~/sorted_data/khan_data_small.csv')
    exercise_index = 'exercise_index'
    json_filename =  os.path.expanduser(
        '~/sorted_data/khan_problem_json_small')
    write_filename =  os.path.expanduser(
        '~/sorted_data/khan_problem_matrix_small.npy')
    process = ProcessData(exercise_filename, exercise_index)
    exercise_data = process.dump_exercise_file_into_json(json_filename)
    process.convert_token_to_matrix(json_data = exercise_data,
                    write_filename = write_filename)
    end = time.time()
    print(end-start)


