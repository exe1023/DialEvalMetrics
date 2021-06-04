"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import numpy as np
from data import Data, ParlAIExtractor
from args import get_args
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils import getbuckets, cosine_similarity
from backup.plot import scatterPlot
from scipy.spatial.distance import euclidean, cityblock
from sklearn.cluster import KMeans
import itertools as it


def temporal_cluster(data: Data):
    """
    Temporal clustering with LDA
    :return:
    """
    X = [t.numpy() for d in data.dial_vecs for t in d if len(d) == 14]
    X = np.stack(X, axis=0)
    ## calculating t-scores for dialog
    Y = []
    for dial in data.dial_vecs:
        if len(dial) == 14:
            for ti in range(len(dial)):
                t_score = int((ti + 1) * 100.0 / float(len(dial)))
                Y.append(t_score)
    lda = LinearDiscriminantAnalysis(n_components=2)
    Y_med_b = getbuckets(Y, 5)
    print(Y_med_b.shape, X.shape)
    # X_r = pca.fit(X).transform(X)
    X_r = lda.fit(X, Y_med_b).transform(X)
    scatterPlot(
        X_r, Y_med_b, 5, N_points=5000, plot_name="tc_{}".format(data.args.data_name)
    )


def semantic_cluster(data: Data):
    """
    Idea: Cluster in Semantic space and measure the change in conversation
    :param data:
    :return:
    """
    labels = []
    tokens = []

    for di, dial in enumerate(data.dial_vecs):
        tokens.extend([d.numpy() for d in dial])
        labels.extend(["{}.{}".format(di, p) for p in range(len(dial))])

    tokens = np.array(tokens)

    kmeans_clustering = KMeans(n_clusters=5)
    idx = kmeans_clustering.fit_predict(tokens)
    clusters = list(range(5))
    cos_distances = {}
    eu_distances = {}
    cb_distances = {}
    for i in it.permutations(clusters, 2):
        cos_distances[i] = cosine_similarity(
            kmeans_clustering.cluster_centers_[i[0]],
            kmeans_clustering.cluster_centers_[i[1]],
        )
        eu_distances[i] = euclidean(
            kmeans_clustering.cluster_centers_[i[0]],
            kmeans_clustering.cluster_centers_[i[1]],
        )
        cb_distances[i] = cityblock(
            kmeans_clustering.cluster_centers_[i[0]],
            kmeans_clustering.cluster_centers_[i[1]],
        )

    # compute
    dist_traversed = []
    flips = []
    prev_di = ""
    dt = 0
    fp = 0
    pc = -1
    for ti, tok in enumerate(tokens):
        di, ui = labels[ti].split(".")
        center = kmeans_clustering.labels_[ti]
        if prev_di == di:
            if pc > -1:
                if pc != center:
                    dt += cos_distances[(pc, center)]
                    fp += 1
        else:
            if dt > 0:
                dist_traversed.append(dt)
                flips.append(fp)
                dt = 0
                fp = 0
            prev_di = di
        pc = center

    print("Flips : Mean : {}, STD: {}".format(np.mean(flips), np.std(flips)))
    print(
        "Distance traversed : Mean : {}, STD : {}".format(
            np.mean(dist_traversed), np.std(dist_traversed)
        )
    )


if __name__ == "__main__":
    args = get_args()
    print("Loading {} data".format(args.data_name))
    data = ParlAIExtractor(args)
    data.load()
    # data.prepare_for_finetuning()
    # plot_tsne(data.dial_vecs, downsample=False, kmeans=False)
    # semantic_cluster(data)
    if args.tc:
        temporal_cluster(data)

