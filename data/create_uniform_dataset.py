'''
    The current dataset is quite diverse and difficult to test on
    so we want to create a more uniform dataset to run on
        for the purpose of this, we will create a dataset
        who spent any time on cc-third-grade-math (520,943)
        this is a relatively uniform set of learners
'''

import time
import os


class CreateUniformData():

    def __init__(self, read_filename,  subject):
        print('exercise file %s' % read_filename)
        self.read_filename = read_filename
        self.subject = subject
        self.subject_learner = set()
        self.exercise_dict = {}

    def find_all_subject_learners(self):
        '''
            iterate through exercise file and create the set of
            learners who spent time on subject
        '''
        reader = open(self.read_filename, 'r')
        print('iterate to find learners')
        first_line = reader.readline().strip()
        col_names = first_line.split(',')
        # locate the subject
        shaid_loc = col_names.index('sha_id')
        exercise_loc = col_names.index('subject')
        counter = 0
        for line in reader:
            line_delimited = line.strip().split(',')
            line_subject = line_delimited[exercise_loc]
            sha_id = line_delimited[shaid_loc]
            if line_subject == self.subject:
                self.subject_learner.add(sha_id)
            # print counter
            counter += 1
            if counter % 1000000 == 0:
                print(counter)
        reader.close()

    def write_uniform_file(self,
                           subjectlearner_filename, subjectonly_filename):
        '''
            iterate through file again
            and write two new files
            <filename>_<subject>learner: contains all learners who spent time on that subject
            <filename>_<subject>only: contain only the learning records with that subject
        '''
        reader = open(self.read_filename, 'r')
        subjectlearner_writer = open(subjectlearner_filename, 'w')
        subjectonly_writer = open(subjectonly_filename, 'w')
        print('iterate through exercise')
        first_line = reader.readline().strip()
        subjectonly_writer.writelines(first_line + '\n')
        subjectlearner_writer.writelines(first_line + '\n')
        col_names = first_line.split(',')
        # locate the subject
        shaid_loc = col_names.index('sha_id')
        exercise_loc = col_names.index('subject')
        counter = 0
        for line in reader:
            line_delimited = line.strip().split(',')
            line_subject = line_delimited[exercise_loc]
            sha_id = line_delimited[shaid_loc]
            if line_subject == self.subject:
                subjectonly_writer.writelines(line)
            if sha_id in self.subject_learner:
                subjectlearner_writer.writelines(line)
            # print counter
            counter += 1
            if counter % 1000000 == 0:
                print(counter)
        reader.close()


def main():
    subject = 'cc-third-grade-math'
    # full read file
    read_filename = os.path.expanduser(
        '~/sorted_data/khan_data_all_sorted.csv')
    uniform_data = CreateUniformData(read_filename, subject)
    uniform_data.find_all_subject_learners()

    # write files for learners and lines in subject set
    subjectlearner_filename = os.path.expanduser(
        '~/sorted_data/khan_data_subjectlearner.csv')
    print(subjectlearner_filename)
    subjectonly_filename = os.path.expanduser(
        '~/sorted_data/khan_data_subjectonly.csv')
    print(subjectonly_filename)
    uniform_data.write_uniform_file(subjectlearner_filename=subjectlearner_filename,
                                    subjectonly_filename=subjectonly_filename)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end-start)
