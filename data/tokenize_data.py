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

    def __init__(self, exercise_filename='', exercise_index_file=''):
        print('exercise file %s' % exercise_filename)
        self.exercise_reader = open(exercise_filename, 'r')
        self.exercise_index_reader = open(exercise_index_file, 'r')
        self.exercise_index = json.load(self.exercise_index_reader)
        self.exercise_data = {}

    def dump_exercise_file_into_json(self, write_filename,
                                     smaller_filename='', tiny_filename=''):
        '''
            iterate through exercise line
            and produce a json file with the
            student id as the key 
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
            counter += 1
            if counter % 1000000 == 0:
                print(counter)
        self.delete_single_session_user()
        # only grab the first 10 students  in the tiny test group (for debugging)
        if tiny_filename != '':
            self.write_smaller_files(tiny_filename, 20)
        # only grab the first 5000 students in the small group (for testing model)
        if smaller_filename != '':
            self.write_smaller_files(smaller_filename, 5000)
        self.write_json_file(write_filename)
        return self.exercise_data

    def delete_single_session_user(self):
        '''
            delete any learner with fewer than 5 sessions
            will not be useful for training
        '''
        students_to_delete = []
        for student in self.exercise_data:
            sessions = len(self.exercise_data[student])
            if sessions < 5:
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
        self.exercise_data[student_id][session_id].append(
            (exercise_id, correct))

    def write_json_file(self, write_filename):
        # store the json file into output
        json_writer = open(write_filename, 'w')
        print('writer file with %f students:' % len(self.exercise_data))
        json.dump(self.exercise_data, json_writer)

    def write_smaller_files(self, small_filename, limit):
        # writer a limited number of students to a smaller file for testing
        small_writer = open(small_filename, 'w')
        small_data = {}
        for i, student in enumerate(self.exercise_data):
            small_data[student] = self.exercise_data[student]
            if i >= limit:
                break
        small_json_writer = open(small_filename, 'w')
        print('writer file with %f students:' % len(small_data))
        json.dump(small_data, small_json_writer)


if __name__ == '__main__':
    start = time.time()
    exercise_filename = os.path.expanduser(
        '~/sorted_data/khan_data_subjectonly.csv')
    exercise_index = 'exercise_index_3only'
    json_filename = os.path.expanduser(
        '~/sorted_data/khan_problem_token_3only')
    small_json_filename = os.path.expanduser(
        '~/sorted_data/khan_problem_token_3only_small')
    tiny_json_filename = os.path.expanduser(
        '~/sorted_data/khan_problem_token_3only_tiny')
    process = ProcessData(exercise_filename, exercise_index)
    exercise_data = process.dump_exercise_file_into_json(json_filename,
                                                         smaller_filename=small_json_filename, tiny_filename=tiny_json_filename)
    end = time.time()
    print(end-start)
