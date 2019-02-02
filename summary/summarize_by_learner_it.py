import numpy as np
import os
import pdb
import csv
import time


class SummarizeLearner():

    def __init__(self, read_filename='', write_filename=''):
        print('initialize ' + read_filename)
        self.reader = open(read_filename, 'r')
        self.user_attempts = {}
        self.write_user_data = {}
        self.last_sha_id = 'sha_id'

    def iterate_through_lines(self):
        '''
            read file and write the lines
            does not use readlines so there's less strain on memory
            it would be great helpful to parallelize this function but
            not sure how to efficiently do this and maintain the sort order 
        '''

        first_line = self.reader.readline().strip()
        col_names = first_line.split(',')
        sha_id_loc = col_names.index('sha_id')
        session_loc = col_names.index('session_start_time')
        # problem_loc = col_names.index('start_time')
        correct_loc = col_names.index('correct')
        attempt_loc = col_names.index('attempt_numbers')
        hint_loc = col_names.index('hints_taken')
        subject_loc = col_names.index('subject')
        topic_loc = col_names.index('topic')
        outgoing_level_loc = col_names.index('outgoing_level')
        counter = 0
        for line in self.reader:
            line_delimited = line.strip().split(',')
            sha_id = line_delimited[sha_id_loc]
            session = line_delimited[session_loc]
            attempt = int(line_delimited[attempt_loc])
            correct = int(line_delimited[correct_loc] == 'true')
            hint = int(line_delimited[hint_loc])
            subject = line_delimited[subject_loc]
            topic = line_delimited[topic_loc]
            outgoing_level = line_delimited[outgoing_level_loc]
            # read each line
            self.parse_line(sha_id, session,  attempt, correct,
                            hint, subject, topic, outgoing_level)
            counter += 1
            if counter % 1000000 == 0:
                print(counter)
        self.reader.close()

    def parse_line(self, sha_id, session,  attempt, correct,
                   hint, subject, topic, outgoing_level):
        '''
           Parse through each line and store the values 
        '''
        if self.last_sha_id == 'sha_id':
            self.last_sha_id = sha_id
            self.user_attempts = {}
            self.update_attempts(session,  attempt, correct,
                                 hint, subject, topic, outgoing_level)
        elif (sha_id != self.last_sha_id) and (self.last_sha_id != 'sha_id'):
            self.summarize_user_data()
            self.last_sha_id = sha_id
            self.user_attempts = {}
            self.update_attempts(session,  attempt, correct,
                                 hint, subject, topic, outgoing_level)
        else:
            self.update_attempts(session,  attempt, correct,
                                 hint, subject, topic, outgoing_level)

    def update_attempts(self, session,  attempt, correct,
                        hint, subject, topic, outgoing_level):
        '''
            update the session meta-data
        '''
        if session not in self.user_attempts:
            self.user_attempts[session] = {}
            self.user_attempts[session]['problem'] = 1
            self.user_attempts[session]['attempt'] = attempt
            self.user_attempts[session]['correct'] = correct
            self.user_attempts[session]['hint'] = hint
            self.user_attempts[session]['subject'] = set(subject)
            self.user_attempts[session]['topic'] = set(topic)
            self.user_attempts[session]['outgoing_level'] = set(outgoing_level)
        else:
            self.user_attempts[session]['problem'] += 1
            self.user_attempts[session]['attempt'] += attempt
            self.user_attempts[session]['correct'] += correct
            self.user_attempts[session]['hint'] += hint
            self.user_attempts[session]['subject'].add(subject)
            self.user_attempts[session]['topic'].add(topic)
            self.user_attempts[session]['outgoing_level'].add(outgoing_level)

    def summarize_user_data(self):
        sha_id = self.last_sha_id
        user_attempts = self.user_attempts
        num_sessions = 0
        num_content = 0
        num_attempts = 0
        num_correct = 0
        num_hint = 0
        num_subject = 0
        num_topic = 0
        num_practiced = 0
        num_mastery1 = 0
        num_mastery2 = 0
        num_mastery3 = 0
        for session in user_attempts:
            num_sessions += 1
            num_content += user_attempts[session]['problem']
            num_attempts += user_attempts[session]['attempt']
            num_correct += user_attempts[session]['correct']
            num_hint += user_attempts[session]['hint']
            num_subject += len(user_attempts[session]['subject'])
            num_topic += len(user_attempts[session]['topic'])
            num_practiced += ('practiced' in
                              user_attempts[session]['outgoing_level'])
            num_mastery1 += ('mastery1' in
                             user_attempts[session]['outgoing_level'])
            num_mastery2 += ('mastery2' in
                             user_attempts[session]['outgoing_level'])
            num_mastery3 += ('mastery3' in
                             user_attempts[session]['outgoing_level'])
        self.store_user_data(
            sha_id, num_sessions, num_content,
            num_attempts, num_correct, num_hint,  num_subject, num_topic,
            num_practiced, num_mastery1, num_mastery2, num_mastery3)

    def store_user_data(self, sha_id, num_sessions, num_content,
                        num_attempts, num_correct, num_hint,  num_subject, num_topic,
                        num_practiced, num_mastery1, num_mastery2, num_mastery3):
        # find percentage metric
        max_session_content = np.max([self.user_attempts[session]['problem']
                                      for session in self.user_attempts])
        perc_correct = num_correct/float(num_content)
        avg_attempts = num_attempts/float(num_content)
        avg_hint = num_hint/float(num_sessions)
        subject_per_session = num_subject / float(num_sessions)
        topic_per_session = num_topic / float(num_sessions)
        perc_session_practiced = num_practiced / float(num_sessions)
        perc_session_mastery1 = num_mastery1 / float(num_sessions)
        perc_session_mastery2 = num_mastery2 / float(num_sessions)
        perc_session_mastery3 = num_mastery3 / float(num_sessions)
        self.write_user_data[sha_id] = [
            sha_id,
            num_sessions,
            num_content,
            max_session_content,
            perc_correct,
            avg_attempts,
            avg_hint,
            subject_per_session,
            topic_per_session,
            perc_session_practiced,
            perc_session_mastery1,
            perc_session_mastery2,
            perc_session_mastery3
        ]

    def write_data(self, write_filename):
        # print(self.writefile)
        with open(write_filename, 'w') as writefile:
            csvwriter = csv.writer(writefile)
            csvwriter.writerow([
                'sha_id',
                'num_sessions',
                'num_content',
                'max_session_content',
                'perc_correct',
                'avg_attempts',
                'avg_hints',
                'subject_per_session',
                'topic_per_session',
                'perc_practiced',
                'perc_mastery1',
                'perc_mastery2',
                'perc_mastery3'
            ])
            for sha_id in self.write_user_data:
                csvwriter.writerow(
                    self.write_user_data[sha_id]
                )


def main():
    read_file = os.path.expanduser(
        '~/sorted_data/khan_data_sorted.csv')
    write_file = os.path.expanduser(
        '~/sorted_data/summarize_khan_data_bylearner_it.csv')
    learner_data = SummarizeLearner(read_filename=read_file,
                                    write_filename=write_file)
    learner_data.iterate_through_lines()
    learner_data.write_data(write_filename=write_file)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end-start)
