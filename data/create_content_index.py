"""
    Create the index for content type and content name
    the index will be used later to populate the one-hot vectors
    used to describe the total 
"""
import json
import numpy as np

class CreateIndex():


    def __init__(self, exercise_filename = '', video_filename = ''):
        print('initialize '+ read_filename)
        self.exercise_reader = open(exercise_filename,'r')
        self.video_reader = open(video_filename,'r')
        self.exercise_set = set()
        self.video_set = set()
        self.exercise_dict = {}
        self.video_dict = {}


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
        for line in self.reader:
            exercise_name = 'exercise:'+line[exercise_loc]
            self.exercise_set.add(exercise_name)
            counter+=1
            if counter % 1000000 == 0:
                print(counter)
        self.exercise_reader.close()

    def iterate_through_video_lines(self):
        '''
            iterate through video file and generate the unique
            set of content
        '''
        first_line = self.video_reader.readline().strip()
        col_names = first_line.split(',')
        video_loc = col_names.index('video_id')
        counter = 0
        for line in self.reader:
            video_name = 'video:'+line[video_loc]
            self.video_set.add(video_name)
            counter+=1
            if counter % 1000000 == 0:
                print(counter)
        self.video_reader.close()


    def sort_and_write_sets(self):
        '''
            store the index for each content item in index
            set the first set item to 1, we define 0 as
            an empty set, no content consumed
        '''
        exercise_array = np.sort([ ex for ex in exercise_set])
        video_array = np.sort([ vid for vid in video_set])
        for i, exercise in enumerate(exercise_array):
            self.exercise_dict[exercise] = i+1
        for i, video in enumerate(video_array):
            self.video_dict[video_dict] = i+1



def main():
    exercise_file = os.path.expanduser( 
        '~/sorted_data/khan_data_sorted.csv')
    video_file = os.path.expanduser( 
        '~/sorted_data/khan_data_video_sorted.csv')
    index_set = CreateIndex(exercise_file, video_file)
    json.dump(index_set.exercise_dict, 'data/exercise_index')
    json.dump(index_set.video_dict, 'data/video_index')

if __name__ == '__main__':
    start = time.time() 
    main()
    end =time.time()
    print(end-start)
