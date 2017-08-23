from __future__ import division
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../weight_learning')
from openravepy import *
import prpy
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
from scipy.stats import multivariate_normal as mvn
#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 4
l1 = 3
l2 = 3
ALPHA = 10
ALPHA_I = 1
ALPHA_O = 1
TRUE_MEAN = np.array([0, 0, 0, 0])
TRUE_WEIGHTS = np.array([4, 3, 2, 1])
TRUE_WEIGHTS = TRUE_WEIGHTS / np.linalg.norm(TRUE_WEIGHTS)
TRUE_WEIGHTS1 = np.array([4, 3])
TRUE_WEIGHTS1 = TRUE_WEIGHTS1 / np.linalg.norm(TRUE_WEIGHTS1)
TRUE_WEIGHTS2 = np.array([2, 1])
TRUE_WEIGHTS2 = TRUE_WEIGHTS2 / np.linalg.norm(TRUE_WEIGHTS2)
NUM_PARTICLES = 500
NUM_ITERATIONS = 25
two_pi = 2 * np.pi
particle_density = 0.01
DEFAULT_DATA_FILE = 'sim_rod_weight_learning.npz'
def plot_feas(data, row=2, col=4, title='Feasible sets to choose from'):
    fig, axes = plt.subplots(nrows=row, ncols=col)
    axes = np.ndarray.flatten(np.array(axes))
    fig2, axes2 = plt.subplots(nrows=row, ncols=col)
    axes2 = np.ndarray.flatten(np.array(axes2))
    for (i, (theta, feasible)) in enumerate(data):
        ax = axes[i]
        ax2 = axes2[i]

        ax.set_xlim(-3, 1.75)
        ax.set_ylim(-3, 1.5)
        ax2.set_xlim(-2, 1)
        ax2.set_ylim(-3, 0.5)

        ax.scatter(feasible[:,0], feasible[:,1], c='C0')
        ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
        ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)

        ax2.scatter(feasible[:,2], feasible[:,3], c='C0')
        ax2.scatter(theta[2], theta[3], c='C2', s=200, zorder=2)
        ax2.scatter(TRUE_MEAN[2], TRUE_MEAN[3], c='C3', s=200, zorder=2)
        plt.pause(0.01)
    fig.suptitle(title + " (dim 1&2)")
    fig2.suptitle(title + " (dim 3&4)")
    plt.pause(0.01)
def get_likelihood(dist, test_set):
    true_likelihoods = 0
    likelihood = dist.neg_log_likelihood(test_set)
    for (theta, feasible) in test_set:
        ll = pe.prob_theta_given_lam_stable_set_weight_num(theta, dist.m, TRUE_WEIGHTS, dist.cost, 1)
        ll -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, dist.m, TRUE_WEIGHTS, dist.cost, 1)
        true_likelihoods += -ll
    return likelihood, true_likelihoods
def train_active(dist, data):
    fig, axes = plt.subplots(nrows=5, ncols=5)
    axes = np.ndarray.flatten(np.array(axes))
    fig2, axes2 = plt.subplots(nrows=5, ncols=5)
    axes2 = np.ndarray.flatten(np.array(axes2))
    bar_fig, bar_axes = plt.subplots(nrows=5, ncols=5)
    bar_axes = np.ndarray.flatten(np.array(bar_axes))
    fig.suptitle('Active Learning dim 1&2')
    fig2.suptitle('Active Learning dim 3&4')

    ground_truth_probs = [utils.prob_of_truth(dist, TRUE_WEIGHTS)]
    ground_truth_dists = [utils.dist_to_truth(dist, TRUE_WEIGHTS)]
    data_likelihoods = [-dist.neg_log_likelihood_mean(test_set)]
    for i in range(1, NUM_ITERATIONS):
        print "\rActive on iteration %d of 24" % i,
        sys.stdout.flush()
        func = partial(info_gain, dist)
        pooled = pool.map(func, data)
        expected_infos = [sample[1] for sample in pooled]
        max_idx = np.argmax(expected_infos)
        (theta, feasible) = pooled[max_idx][0]
        actual_infos = []
        ent_before = dist.entropy(num_boxes=20)
        for j in range(len(data)):
            t, f = data[j]
            d = SetMeanParticleDistribution(dist.particles, dist.weights, dist.cost, dist.m, dist.ALPHA_I, dist.ALPHA_O, dist.h)
            d.weights = d.reweight(t, f)
            actual_infos.append(ent_before - d.entropy(num_boxes=20))

        dist.weights = dist.reweight(theta, feasible)
        dist.resample()

        particles1 = dist.particles[:,:2] / np.linalg.norm(dist.particles[:,:2], axis=1).reshape(-1, 1)
        particles2 = dist.particles[:,2:] / np.linalg.norm(dist.particles[:,2:], axis=1).reshape(-1, 1)
        utils.plot_belief(axes[i], particles1, TRUE_WEIGHTS1, particle_density)
        utils.plot_belief(axes2[i], particles2, TRUE_WEIGHTS2, particle_density)
        bar_axes[i].bar(np.arange(len(data)), expected_infos, 0.35, color='C0', label='expected info gain')
        bar_axes[i].bar(np.arange(len(data)) + 0.35, actual_infos, 0.35, color='C1', label='actual info gain')
        bar_axes[i].bar(max_idx, expected_infos[max_idx], 0.35, color='C2', label='chosen set expected info')

        prob = utils.prob_of_truth(dist, TRUE_WEIGHTS)
        distance = utils.dist_to_truth(dist, TRUE_WEIGHTS)
        ll = dist.neg_log_likelihood_mean(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
        plt.pause(0.01)
    print
    return ground_truth_probs, ground_truth_dists, data_likelihoods
def train_min_cost(dist, data):
    fig, axes = plt.subplots(nrows=5, ncols=5)
    axes = np.ndarray.flatten(np.array(axes))
    fig2, axes2 = plt.subplots(nrows=5, ncols=5)
    axes2 = np.ndarray.flatten(np.array(axes2))
    bar_fig, bar_axes = plt.subplots(nrows=5, ncols=5)
    bar_axes = np.ndarray.flatten(np.array(bar_axes))
    fig.suptitle('Min Cost dim 1&2')
    fig2.suptitle('Min Cost dim 3&4')

    ground_truth_probs = [utils.prob_of_truth(dist, TRUE_WEIGHTS)]
    ground_truth_dists = [utils.dist_to_truth(dist, TRUE_WEIGHTS)]
    data_likelihoods = [-dist.neg_log_likelihood_mean(test_set)]
    for i in range(1, NUM_ITERATIONS):
        print "\rMin cost on iteration %d of 24" % i,
        sys.stdout.flush()
        func = partial(min_cost, dist)
        pooled = pool.map(func, data)
        expected_costs = [sample[2] for sample in pooled]
        max_idx = np.argmin(expected_costs)
        (theta, feasible) = pooled[max_idx][0]
        actual_infos = []
        ent_before = dist.entropy(num_boxes=20)
        for j in range(len(data)):
            t, f = data[j]
            d = SetMeanParticleDistribution(dist.particles, dist.weights, dist.cost, dist.m, dist.ALPHA_I, dist.ALPHA_O, dist.h)
            d.weights = d.reweight(t, f)
            actual_infos.append(ent_before - d.entropy(num_boxes=20))

        dist.weights = dist.reweight(theta, feasible)
        dist.resample()

        particles1 = dist.particles[:,:2] / np.linalg.norm(dist.particles[:,:2], axis=1).reshape(-1, 1)
        particles2 = dist.particles[:,2:] / np.linalg.norm(dist.particles[:,2:], axis=1).reshape(-1, 1)
        utils.plot_belief(axes[i], particles1, TRUE_WEIGHTS1, particle_density)
        utils.plot_belief(axes2[i], particles2, TRUE_WEIGHTS2, particle_density)
        bar_axes[i].bar(np.arange(len(data)) + 0.35, actual_infos, 0.35, color='C1')
        bar_axes[i].bar(max_idx, expected_costs[max_idx], 0.35, color='C2')

        prob = utils.prob_of_truth(dist, TRUE_WEIGHTS)
        distance = utils.dist_to_truth(dist, TRUE_WEIGHTS)
        ll = dist.neg_log_likelihood_mean(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
        plt.pause(0.01)
    print
    return ground_truth_probs, ground_truth_dists, data_likelihoods
def train_random(dist, data):
    fig, axes = plt.subplots(nrows=5, ncols=5)
    axes = np.ndarray.flatten(np.array(axes))
    fig2, axes2 = plt.subplots(nrows=5, ncols=5)
    axes2 = np.ndarray.flatten(np.array(axes2))
    fig.suptitle('Random dim 1&2')
    fig2.suptitle('Random dim 3&4')

    ground_truth_probs = [utils.prob_of_truth(dist, TRUE_WEIGHTS)]
    ground_truth_dists = [utils.dist_to_truth(dist, TRUE_WEIGHTS)]
    data_likelihoods = [-dist.neg_log_likelihood_mean(test_set)]
    for i in range(1, NUM_ITERATIONS):
        print "\rRandom on iteration %d of 24" % i,
        sys.stdout.flush()
        (theta, feasible) = data[i % 8]
        dist.weights = dist.reweight(theta, feasible)
        dist.resample()

        particles1 = dist.particles[:,:2] / np.linalg.norm(dist.particles[:,:2], axis=1).reshape(-1, 1)
        particles2 = dist.particles[:,2:] / np.linalg.norm(dist.particles[:,2:], axis=1).reshape(-1, 1)
        utils.plot_belief(axes[i], particles1, TRUE_WEIGHTS1, particle_density)
        utils.plot_belief(axes2[i], particles2, TRUE_WEIGHTS2, particle_density)

        prob = utils.prob_of_truth(dist, TRUE_WEIGHTS)
        distance = utils.dist_to_truth(dist, TRUE_WEIGHTS)
        ll = dist.neg_log_likelihood_mean(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
        plt.pause(0.01)
    print
    return ground_truth_probs, ground_truth_dists, data_likelihoods
#########################################################
datafile = utils.prefilled_input('Simulation data file: ', DEFAULT_DATA_FILE)

f = np.load('../data/' + datafile)
# idxs = list(range(25))
idxs = [2, 5, 9, 18, 1, 11, 14, 23]
# idxs = np.random.choice(len(f['data']), size=25)
print idxs
# data_full = f['data_full'][idxs]
all_data = f['data'][idxs]
test_set = f['data'][25:35]
# poses = f['poses'][idxs]
data = all_data

for (i, (theta, feasible)) in enumerate(data):
    if len(feasible) > 1000:
        idxs = np.random.choice(len(feasible), size=1000)
        new_feas = feasible[idxs]
        data[i][1] = new_feas
data_active = data[:8]
# test_set = data_active
plot_feas(test_set, 2, 5, 'Test feasible sets')
plot_feas(data_active)
fig, axes = plt.subplots(nrows=2, ncols=4)
axes = np.ndarray.flatten(np.array(axes))
fig2, axes2 = plt.subplots(nrows=2, ncols=4)
axes2 = np.ndarray.flatten(np.array(axes2))
for (i, (theta, feasible)) in enumerate(data_active):
    utils.plot_likelihood_heatmap_norm_weights(axes[i], theta[:2], feasible[:,:2], \
    TRUE_WEIGHTS1, vmin=-7.5, vmax=-2)
    utils.plot_likelihood_heatmap_norm_weights(axes2[i], theta[2:], feasible[:,2:], \
    TRUE_WEIGHTS2, vmin=-7.5, vmax=-2)
fig.suptitle('likelihoods dim 1&2')
fig2.suptitle('likelihoods dim 3&4')

particles = []
weights = []
while len(particles) < NUM_PARTICLES:
    p = np.random.randn(DOF, 1).T[0]
    p = p / np.linalg.norm(p, axis=0)
    if p[0] >= 0 and p[1] >= 0 and p[2] >= 0 and p[3] >= 0:
        particles.append(p)
particles = np.array(particles)
weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

dist_active = SetMeanParticleDistribution(particles, weights, utils.cost, m=TRUE_MEAN, \
ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=0.01)
dist_min_cost = SetMeanParticleDistribution(np.copy(particles), np.copy(weights), utils.cost, m=TRUE_MEAN, \
ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=0.01)
dist_random = SetMeanParticleDistribution(np.copy(particles), np.copy(weights), utils.cost, m=TRUE_MEAN, \
ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=0.01)

initial_prob = utils.prob_of_truth(dist_active, TRUE_WEIGHTS)
initial_dist = utils.dist_to_truth(dist_active, TRUE_WEIGHTS)
initial_ll = -dist_active.neg_log_likelihood_mean(test_set)

def info_gain(dist, x):
    return (x, dist.info_gain(x[1], num_boxes=20), dist.expected_cost(x[1]))
def min_cost(dist, x):
    return (x, 0, dist.expected_cost(x[1]))
if __name__ == '__main__':
    pool = mp.Pool(8)

    probs_active, dists_active, ll_active = train_active(dist_active, data_active)
    print "finished active"
    probs_min_cost, dists_min_cost, ll_min_cost = train_min_cost(dist_min_cost, data_active)
    print "finished passive"
    probs_random, dists_random, ll_random = train_random(dist_random, data_active)
    print "finished random"

    probs_active[0] = initial_prob
    probs_min_cost[0] = initial_prob
    probs_random[0] = initial_prob

    dists_active[0] = initial_dist
    dists_min_cost[0] = initial_dist
    dists_random[0] = initial_dist

    ll_active[0] = initial_ll
    ll_min_cost[0] = initial_ll
    ll_random[0] = initial_ll

    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(probs_random, label='randomly selected')
    ax.plot(probs_min_cost, label='min cost')
    ax.plot(probs_active, label='active learning')
    ax.legend(loc='upper left')
    ax.set_xlabel('iteration')
    ax.set_ylabel('probability density at ground truth')
    plt.pause(0.1)

    ax = fig.add_subplot(132)
    ax.plot(dists_random, label='randomly selected')
    ax.plot(dists_min_cost, label='min cost')
    ax.plot(dists_active, label='active learning')
    ax.legend(loc='upper left')
    ax.set_xlabel('iteration')
    ax.set_ylabel('distance of mean to ground truth')
    plt.pause(0.1)

    ax = fig.add_subplot(133)
    ax.plot(ll_random, label='randomly selected')
    ax.plot(ll_min_cost, label='min cost')
    ax.plot(ll_active, label='active learning')
    ax.legend(loc='upper left')
    ax.set_xlabel('iteration')
    ax.set_ylabel('log likelihood of test set')
    plt.pause(0.1)
    plt.show()