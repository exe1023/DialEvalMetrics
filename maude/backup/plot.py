"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

PLOTS_DIR = "plots"


def scatterPlot(X, Y, n_classes, N_points=100, plot_name="scatter_plot"):
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)
    indices = random.sample(range(max(Y.shape)), N_points)
    colors = cm.rainbow(np.linspace(0, 1, n_classes))
    for i in indices:
        plt.scatter(X[i, 0], X[i, 1], color=colors[int(Y[i])], label=int(Y[i]))
    plt.savefig(PLOTS_DIR + "/{}.png".format(plot_name))


def plot_tsne(
    dialogs,
    mode="all",
    alg="tsne",
    dim=50,
    hd=1,
    downsample=True,
    kmeans=True,
    plot_name="tsne_plot",
    N_points=2000,
):
    """
    Creates and TSNE model and plots it
    mode - all, print both train and test words
    """
    labels = []
    tokens = []

    for di, dial in enumerate(dialogs):
        tokens.extend([d.numpy() for d in dial])
        labels.extend(["{}.{}".format(di, p) for p in range(len(dial))])

    tokens = np.array(tokens)
    # Select a subsample
    indices = random.sample(range(max(tokens.shape)), N_points)
    tokens = tokens[indices]
    print(tokens.shape)

    if downsample:
        # first downsample the model using pca to dim
        down_model = PCA(n_components=dim, whiten=True)
        tokens = down_model.fit(tokens).transform(tokens)
        print(tokens.shape)

    if kmeans:
        # semantic clustering
        kmeans_clustering = KMeans(n_clusters=5)
        idx = kmeans_clustering.fit_predict(tokens)

    if alg == "tsne":
        model = TSNE(
            perplexity=10, n_components=2, init="random", n_iter=2000, random_state=23
        )
        new_values = model.fit_transform(tokens)
    else:
        model = PCA(n_components=2, whiten=True)
        new_values = model.fit(tokens).transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    colors = np.array([x for x in "bgrcmykbgrcmykbgrcmykbgrcmyk"])
    for i in range(len(x)):
        color = "r"
        if kmeans:
            color = colors[kmeans_clustering.labels_[i]]
        else:
            # temporal
            dial_t = int(labels[i].split(".")[-1])
            if dial_t < 5:
                color = "r"
            elif dial_t < 10:
                color = "g"
            else:
                color = "b"

        plt.scatter(x[i], y[i], c=color)
        if int(labels[i].split(".")[0]) == hd:
            plt.annotate(
                labels[i],
                xy=(x[i], y[i]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
            )

    plt.savefig(PLOTS_DIR + "/{}.png".format(plot_name))

