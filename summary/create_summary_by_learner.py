"""Summarize the input session activity data by learner. Transform the input
data which has one row per activity, session, learner into a summary with one 
row by learner. Include metrics like: 
    number of sessions, number of content, avg correct on content, highest
    mastery level reached
"""

import pandas as pd
import numpy as np
import pdb
import time


begin = time.time()
# read in data
# [TODO] change directory for larger data
data = pd.read_csv('~/sorted_data/khan_data_small.csv')

# add exercises
data['is_practiced'] = (data['outgoing_level'] == 'practiced').astype(int)
data['is_mastery1'] = (data['outgoing_level'] == 'mastery1').astype(int)
data['is_mastery2'] = (data['outgoing_level'] == 'mastery2').astype(int)
data['is_mastery3'] = (data['outgoing_level'] == 'mastery3').astype(int)


# group by session
data_by_session = data.groupby(['sha_id', 'session_start_time']).agg({
    # number of problem content
    'start_time': np.count_nonzero,
    # number of correct responses on problem content
    'correct': np.sum,
    # number of hints used
    'hints_taken': np.sum,
    # number of attempts on problem
    'attempt_numbers': np.sum,
    # subject
    'subject': pd.Series.nunique,
    # topic
    'topic': pd.Series.nunique,
    # is_practiced
    'is_practiced': np.max,
    # is_mastery1
    'is_mastery1': np.max,
    # is_mastery2
    'is_mastery2': np.max,
    # is_mastery3
    'is_mastery3': np.max
}).reset_index()

print('grouped by sessions')

# group by learners
data_by_learner = data_by_session.groupby('sha_id').agg({
    # number of sessions
    'session_start_time': np.count_nonzero,
    # number of problem content
    'start_time': np.sum,
    # number of exercises
    'correct': np.sum,
    # number of attempts on problem
    'attempt_numbers': np.sum,
    # session subjects
    'subject': np.sum,
    # topic subjects
    'topic': np.sum,
    # is_practiced
    'is_practiced': np.max,
    # is_mastery1
    'is_mastery1': np.max,
    # is_mastery2
    'is_mastery2': np.max,
    # is_mastery3
    'is_mastery3': np.max
})

data_by_learner = data_by_learner.rename(index=str, columns={
    'session_start_time': 'num_sessions',
    'start_time':  'num_content',
    'correct':   'num_correct',
    'attempt_numbers':   'num_attempts',
    'subject':   'num_subject_sessions',
    'topic':   'num_topic_sessions'})

data_by_learner['subject_per_session'] = (
    data_by_learner['num_subject_sessions']/data_by_learner['num_sessions'])
data_by_learner['topic_per_session'] = (
    data_by_learner['num_topic_sessions']/data_by_learner['num_sessions'])

# select columsn to write
data_to_write = data_by_learner[[
    'num_sessions',
    'num_content',
    'num_correct',
    'num_attempts',
    'subject_per_session',
    'topic_per_session',
    'is_practiced',
    'is_mastery1',
    'is_mastery2',
    'is_mastery3']]


data_to_write.to_csv('~/sorted_data/khan_problem_data_by_learner_it.csv')


end = time.time()
print('%s seconds to run' % (end - begin))
