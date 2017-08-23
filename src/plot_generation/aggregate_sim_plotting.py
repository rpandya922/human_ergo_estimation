from __future__ import division
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../weight_learning')
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde as kde
from sklearn.decomposition import PCA
import probability_estimation as pe
from distribution import SetMeanParticleDistribution
import multiprocessing as mp
from functools import partial
from random import shuffle
from sklearn.neighbors import NearestNeighbors
import utils
import readline
import argparse
import pickle
from scipy.stats import multivariate_normal as mvn

ground_truth_probs_active = []
ground_truth_dists_active = []
data_likelihoods_active = []

ground_truth_probs_passive = []
ground_truth_dists_passive = []
data_likelihoods_passive = []

ground_truth_probs_random = []
ground_truth_dists_random = []
data_likelihoods_random = []

for set_idx in range(10):
    for param_idx in range(5):
        try:
            pkl_file = open('../avg_data/set%s_param%s.pkl' % (set_idx, param_idx), 'rb')
        except:
            print 'failed'
            continue
        data = pickle.load(pkl_file)

        initial_prob = data['initial_prob']
        initial_dist = data['initial_dist']
        initial_ll = data['initial_ll']

        probs_active = data['probs_active']
        probs_passive = data['probs_passive']
        probs_random = data['probs_random']

        dists_active = data['distances_active']
        dists_passive = data['distances_passive']
        dists_random = data['distances_random']

        ll_active = data['ll_active']
        ll_passive = data['ll_passive']
        ll_random = data['ll_random']

        probs_active[0] = initial_prob
        probs_passive[0] = initial_prob
        probs_random[0] = initial_prob
        ground_truth_probs_active.append(probs_active)
        ground_truth_probs_passive.append(probs_passive)
        ground_truth_probs_random.append(probs_random)

        dists_active[0] = initial_dist
        dists_passive[0] = initial_dist
        dists_random[0] = initial_dist
        ground_truth_dists_active.append(dists_active)
        ground_truth_dists_passive.append(dists_passive)
        ground_truth_dists_random.append(dists_random)

        ll_active[0] = initial_ll
        ll_passive[0] = initial_ll
        ll_random[0] = initial_ll
        data_likelihoods_active.append(ll_active)
        data_likelihoods_passive.append(ll_passive)
        data_likelihoods_random.append(ll_random)

probs_random = np.average(ground_truth_probs_random, axis=0)
probs_passive = np.average(ground_truth_probs_passive, axis=0)
probs_active = np.average(ground_truth_probs_active, axis=0)
fig = plt.figure()
ax = fig.add_subplot(131)
ax.plot(probs_random, label='randomly selected')
ax.plot(probs_passive, label='passive learning')
ax.plot(probs_active, label='active learning')
ax.legend(loc='upper left')
ax.set_xlabel('iteration')
ax.set_ylabel('probability density at ground truth')
plt.pause(0.1)

dists_random = np.average(ground_truth_dists_random, axis=0)
dists_passive = np.average(ground_truth_dists_passive, axis=0)
dists_active = np.average(ground_truth_dists_active, axis=0)
ax = fig.add_subplot(132)
ax.plot(dists_random, label='randomly selected')
ax.plot(dists_passive, label='passive learning')
ax.plot(dists_active, label='active learning')
ax.legend(loc='upper left')
ax.set_xlabel('iteration')
ax.set_ylabel('distance of mean to ground truth')
plt.pause(0.1)

ll_random = np.average(data_likelihoods_random, axis=0)
ll_passive = np.average(data_likelihoods_passive, axis=0)
ll_active = np.average(data_likelihoods_active, axis=0)
ax = fig.add_subplot(133)
ax.plot(ll_random, label='randomly selected')
ax.plot(ll_passive, label='passive learning')
ax.plot(ll_active, label='active learning')
ax.legend(loc='upper left')
ax.set_xlabel('iteration')
ax.set_ylabel('log likelihood of test set')
plt.pause(0.1)
plt.show()