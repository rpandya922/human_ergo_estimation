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

DATA_LOCATION = '../data/mode_data'

def calc_error_bounds(data):
    n = np.array(data).shape[1]
    sample_mean = np.mean(data, axis=0)
    sample_std = np.std(data, axis=0)
    return (1 / np.sqrt(n)* sample_std)
def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)
def plot_info_bars(expected_infos, actual_infos):
    bar_fig, bar_axes = plt.subplots(nrows=2, ncols=5)
    bar_axes = np.ndarray.flatten(np.array(bar_axes))
    for i, infos in enumerate(expected_infos):
        ax = bar_axes[i]
        max_idx = np.argmax(infos)
        ax.bar(np.arange(len(infos)), infos, 0.35, color='C0', label='expected info gain')
        ax.bar(np.arange(len(infos)) + 0.35, actual_infos[i], 0.35, color='C1', label='actual info gain')
        ax.bar(max_idx, infos[max_idx], 0.35, color='C2', label='chosen set expected info')
        plt.pause(0.01)
def plot_cost_bars(expected_costs):
    bar_fig, bar_axes = plt.subplots(nrows=2, ncols=5)
    bar_axes = np.ndarray.flatten(np.array(bar_axes))
    for i, costs in enumerate(expected_costs):
        ax = bar_axes[i]
        min_idx = np.argmin(costs)
        ax.bar(np.arange(len(costs)), costs, 0.35, color='C0', label='expected cost')
        ax.bar(min_idx, costs[min_idx], 0.35, color='C2', label='chosen set expected cost')
    plt.pause(0.01)
def plot_final_belief(dist, weights):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    fig2 = plt.figure()
    axes2 = fig2.add_subplot(111)
    weights1 = weights[:2] / np.linalg.norm(weights[:2])
    weights2 = weights[2:4] / np.linalg.norm(weights[2:4])
    particles1 = dist.particles[:,:2] / np.linalg.norm(dist.particles[:,:2], axis=1).reshape(-1, 1)
    particles2 = dist.particles[:,2:4] / np.linalg.norm(dist.particles[:,2:4], axis=1).reshape(-1, 1)
    utils.plot_belief(axes, particles1, weights1, 0.008)
    utils.plot_belief(axes2, particles2, weights2, 0.008)
    plt.show(0.01)

ground_truth_probs_active = []
ground_truth_dists_active = []
data_likelihoods_active = []

ground_truth_probs_passive = []
ground_truth_dists_passive = []
data_likelihoods_passive = []

ground_truth_probs_random = []
ground_truth_dists_random = []
data_likelihoods_random = []
test_set = []
weights = []
for set_idx in range(10):
    for param_idx in range(5):
        try:
            pkl_file = open('%s/set%s_param%s.pkl' % (DATA_LOCATION, set_idx, param_idx), 'rb')
        except:
            continue
        data = pickle.load(pkl_file)
        test_set = data['test_set']
        weights = data['weights']
        # plot_info_bars(data['expected_infos'], data['actual_infos'])
        # plot_cost_bars(data['expected_costs'])
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

x = list(range(10))

probs_random = np.mean(ground_truth_probs_random, axis=0)
probs_passive = np.mean(ground_truth_probs_passive, axis=0)
probs_active = np.mean(ground_truth_probs_active, axis=0)
error_probs_random = calc_error_bounds(ground_truth_probs_random)
error_probs_passive = calc_error_bounds(ground_truth_probs_passive)
error_probs_active = calc_error_bounds(ground_truth_probs_active)
fig = plt.figure()
ax = fig.add_subplot(131)
ax.set_xlim(0, 4)
ax.plot(probs_random, label='randomly selected')
ax.plot(probs_passive, label='passive learning')
ax.plot(probs_active, label='active learning')
ax.fill_between(x, probs_random - error_probs_random, probs_random + error_probs_random, alpha=0.3)
ax.fill_between(x, probs_passive - error_probs_passive, probs_passive + error_probs_passive, alpha=0.3)
ax.fill_between(x, probs_active - error_probs_active, probs_active + error_probs_active, alpha=0.3)
ax.legend(loc='upper left')
ax.set_xlabel('iteration')
ax.set_ylabel('probability density at ground truth')
plt.pause(0.1)

dists_random = np.mean(ground_truth_dists_random, axis=0)
dists_passive = np.mean(ground_truth_dists_passive, axis=0)
dists_active = np.mean(ground_truth_dists_active, axis=0)
error_dists_random = calc_error_bounds(ground_truth_dists_random)
error_dists_passive = calc_error_bounds(ground_truth_dists_passive)
error_dists_active = calc_error_bounds(ground_truth_dists_active)
ax = fig.add_subplot(132)
ax.set_xlim(0, 4)
ax.plot(dists_random, label='randomly selected')
ax.plot(dists_passive, label='passive learning')
ax.plot(dists_active, label='active learning')
ax.fill_between(x, dists_random - error_dists_random, dists_random + error_dists_random, alpha=0.3)
ax.fill_between(x, dists_passive - error_dists_passive, dists_passive + error_dists_passive, alpha=0.3)
ax.fill_between(x, dists_active - error_dists_active, dists_active + error_dists_active, alpha=0.3)
ax.legend(loc='upper left')
ax.set_xlabel('iteration')
ax.set_ylabel('distance of mode to ground truth')
plt.pause(0.1)

def neg_log_likelihood(data, ground_truth):
    total = 0
    for (theta, feasible) in data:
        total += pe.prob_theta_given_lam_stable_set_weight_num(theta, np.zeros(7), ground_truth, cost, 1)
        total -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, np.zeros(7), ground_truth, cost, 1)
    return -total

error_ll_random = calc_error_bounds(data_likelihoods_random)
error_ll_passive = calc_error_bounds(data_likelihoods_passive)
error_ll_active = calc_error_bounds(data_likelihoods_active)
ll_random = np.mean(data_likelihoods_random, axis=0)
ll_passive = np.mean(data_likelihoods_passive, axis=0)
ll_active = np.mean(data_likelihoods_active, axis=0)
ax = fig.add_subplot(133)
ax.set_xlim(0, 4)
ax.plot(ll_random, label='randomly selected')
ax.plot(ll_passive, label='passive learning')
ax.plot(ll_active, label='active learning')
ax.plot([-neg_log_likelihood(test_set, weights)]*10)
ax.fill_between(x, ll_random - error_ll_random, ll_random + error_ll_random, alpha=0.3)
ax.fill_between(x, ll_passive - error_ll_passive, ll_passive + error_ll_passive, alpha=0.3)
ax.fill_between(x, ll_active - error_ll_active, ll_active + error_ll_active, alpha=0.3)
ax.legend(loc='upper left')
ax.set_xlabel('iteration')
ax.set_ylabel('log likelihood of test set')
plt.pause(0.1)
plt.show()