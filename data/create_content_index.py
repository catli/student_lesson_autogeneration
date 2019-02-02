"""
    Create the index for content type and content name
    the index will be used later to populate the one-hot vectors
    used to describe the total 
"""
import json
import numpy as np
import time
import os


class CreateIndex():

    def __init__(self, exercise_filename='', video_filename=''):
        print('exercise file %s' % exercise_filename)
        print('video file %s' % video_filename)
        self.exercise_reader = open(exercise_filename, 'r')
        # [TODO] UNCOMMENT FOR VIDEO
        #self.video_reader = open(video_filename,'r')
        self.exercise_set = set()
        self.video_set = set()
        self.exercise_dict = {}
        self.video_dict = {}

    def create_dict(self):
        self.iterate_through_exercise_lines()
        # [TODO] UNCOMMENT FOR VIDEO
        # self.iterate_through_video_lines()
        self.sort_and_write_exercise_sets()
        # self.sort_and_write_video_sets()

    def iterate_through_exercise_lines(self):
        '''
            iterate through exercise file and generate the unique
            set of content
        '''
        print('iterate through exercise')
        first_line = self.exercise_reader.readline().strip()
        col_names = first_line.split(',')
        exercise_loc = col_names.index('exercise')
        counter = 0
        for line in self.exercise_reader:
            line_delimited = line.strip().split(',')
            exercise_name = 'exercise:'+line_delimited[exercise_loc]
            self.exercise_set.add(exercise_name)
            counter += 1
            if counter % 1000000 == 0:
                print(counter)
        self.exercise_reader.close()

    def iterate_through_video_lines(self):
        '''
            iterate through video file and generate the unique
            set of content
        '''
        print('iterate through video')
        first_line = self.video_reader.readline().strip()
        col_names = first_line.split(',')
        video_loc = col_names.index('video_id')
        counter = 0
        for line in self.video_reader:
            line_delimited = line.strip().split(',')
            video_name = 'video:'+line_delimited[video_loc]
            self.video_set.add(video_name)
            counter += 1
            if counter % 1000000 == 0:
                print(counter)
        self.video_reader.close()

    def sort_and_write_exercise_sets(self):
        '''
            store the index for each content item in index
            set the first set item to 1, we define 0 as
            an empty set, no content consumed
        '''
        exercise_array = np.sort([ex for ex in self.exercise_set])
        for i, exercise in enumerate(exercise_array):
            self.exercise_dict[exercise] = i+1

    def sort_and_write_video_sets(self):
        '''
            store the index for each content item in index
            set the first set item to 1, we define 0 as
            an empty set, no content consumed
        '''
        video_array = np.sort([vid for vid in self.video_set])
        for i, video in enumerate(video_array):
            self.video_dict[video] = i+1


def main():
    exercise_file = os.path.expanduser(
        '~/sorted_data/khan_data_subjectlearner.csv')
    video_file = os.path.expanduser(
        '~/sorted_data/khan_video_data_sorted.csv')
    index_set = CreateIndex(exercise_file, video_file)
    index_set.create_dict()
    exercise_writer = open('exercise_index_3learner', 'w')
    # [TODO] UNCOMMENT TO RUN VIDEO FILE
    # video_writer = open('video_index','w')
    json.dump(index_set.exercise_dict, exercise_writer)
    # json.dump(index_set.video_dict, video_writer )


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end-start)
