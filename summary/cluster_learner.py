from collections import Counter
"""
    Use kmeans-cluster to group learners by metric
    and find cutpoints for each group
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import json
import pdb

from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



def unscale_center(center, prescale_std, prescale_mean):
    unscaled_center = (center * prescale_std + prescale_mean)
    return unscaled_center


def drop_id_columns(data):
    # drop the id column for scale
    data_without_id = data.drop(labels='sha_id', axis=1)
    return data_without_id


def find_prescale_metric(data_without_id):
    # return the mean and std of data before scaling
    prescale_mean = np.array(data_without_id.mean(axis=0))
    prescale_std = np.array(data_without_id.std(axis=0))
    return prescale_mean, prescale_std


def create_scaled_data(data_without_id):
    # scale data by feature, normalized to mean and variance
    scaled_data = scale(data_without_id)
    return scaled_data


def calculate_labels_and_centers(scaled_data):
    kmeans = KMeans(n_clusters=3, random_state=0).fit(scaled_data)
    # write unscaled centers
    return kmeans


def create_scaled_centers(kmeans, data_without_id):
    # show the kmeans center from the
    # calculated kmeans using unscaled number
    prescale_mean, prescale_std = find_prescale_metric(data_without_id)
    data_columns = data_without_id.columns
    unscaled_centers = [
        unscale_center(array,
                       prescale_std, prescale_mean)
        for array in kmeans.cluster_centers_]
    centers_df = pd.DataFrame(unscaled_centers)
    centers_df.columns = data_columns
    return centers_df


def calculate_pca(data_without_id):
    # reduce data into 2 dimension to visualize cluster
    pca = PCA(n_components=2).fit(data_without_id)
    pca_data = pca.transform(data_without_id)
    return pca_data


def visualize_cluster(kmeans, scaled_data):
    # visualize the pca of each kmeans cluster
    # using the scaled dataset
    pca_data = calculate_pca(scaled_data)
    label_set = max(kmeans.labels_)
    color = []
    for label in kmeans.labels_:
        if label == 0:
            color.append('yellow')
        elif label == 1:
            color.append('blue')
        elif label == 2:
            color.append('gray')
    plt.plot(pca_data[:, 0], pca_data[:, 1], 'k.', markersize=0)
    plt.scatter(pca_data[:, 0], pca_data[:, 1],
                marker='o', s=1, c=color)
    plt_path = os.path.expanduser('~/sorted_data/khan_learner_cluster.png')
    plt.savefig(plt_path)
    plt.close()


def create_labels_output(data, kmeans, label_path):
    # dump the json labels into a file
    sha_ids = data['sha_id']
    labels = kmeans.labels_
    # print the count per group
    print(Counter(labels))
    label_json = {}
    for i, label in enumerate(labels):
        sha_id = sha_ids[i]
        label_json[sha_id] = str(label)
    json_writefile = open(label_path, 'w')
    json.dump(label_json, json_writefile)
    json_writefile.close()
    return label_json


def calculate_kmeans():
    read_path = os.path.expanduser(
        '~/sorted_data/summarize_khan_data_bylearner.csv')
    data = pd.read_csv(read_path)
    data_without_id = drop_id_columns(data)
    scaled_data = create_scaled_data(data_without_id)
    kmeans = calculate_labels_and_centers(scaled_data)
    visualize_cluster(kmeans, scaled_data)
    label_path = os.path.expanduser('~/sorted_data/khan_learner_kmean_labels')
    label_json = create_labels_output(data, kmeans, label_path)
    centers_df = create_scaled_centers(kmeans, data_without_id)
    centers_df.to_csv('~/sorted_data/khan_learner_group_centers.csv')
    return kmeans.labels_, centers_df


if __name__ == '__main__':
    start = time.time()
    labels, centers = calculate_kmeans()
    end = time.time()
    print(end-start)
