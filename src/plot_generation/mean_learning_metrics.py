from __future__ import division
import sys
sys.path.insert(0, '../')
from openravepy import *
import prpy
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde as kde
from sklearn.decomposition import PCA
import probability_estimation as pe
from distribution import SetWeightsParticleDistribution
import multiprocessing as mp
from functools import partial
from random import shuffle
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal as mvn
#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 4
NUM_PARTICLES = 500
box_size = 0.5
ALPHA_I = 2.5
ALPHA_O = 2.5
TRUE_MEAN = np.array([0, 0, 0, 0])
TRUE_WEIGHTS = np.array([1, 1, 1, 1])
all_vars_dim1 = []
all_vars_dim2 = []
def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)
def neg_log_likelihood(data, mean):
    total = 0
    for (theta, feasible) in data:
        total += pe.prob_theta_given_lam_stable_set_weight_num(theta, mean, TRUE_WEIGHTS, cost, 1)
        total -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, mean, TRUE_WEIGHTS, cost, 1)
    return -total
def plot_feas(data):
    fig, axes = plt.subplots(nrows=4, ncols=4)
    axes = np.ndarray.flatten(np.array(axes))
    fig2, axes2 = plt.subplots(nrows=4, ncols=4)
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
        plt.pause(0.2)
    fig.suptitle("Feasible sets to choose from (dim 1&2)")
    fig2.suptitle("Feasible sets to choose from (dim 3&4)")
    plt.pause(0.2)
def plot_belief(ax, particles, ground_truth, second=False):
    data_means = particles.T
    kernel = kde(data_means)
    if second:
        xx, yy = np.mgrid[-2:1:100j, -3:0.5:100j]
    else:
        xx, yy = np.mgrid[-3:1.75:100j, -3:1.5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='Greens')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.scatter(ground_truth[0], ground_truth[1], c='C3', s=200, zorder=2)
def plot_belief_update(ax, particles, theta, feasible, ground_truth, second=False):
    plot_belief(ax, particles, ground_truth, second)
    ax.scatter(feasible[:,0], feasible[:,1], c='C0')
    ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
def plot_likelihood_heatmap(ax, theta, feasible, ground_truth, second=False, with_belief=False, dist=None):
    if second:
        xx, yy = np.mgrid[-2:1:100j, -3:0.5:100j]
    else:
        xx, yy = np.mgrid[-3:1.75:100j, -3:1.5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feasible)
    distances, indices = nbrs.kneighbors(feasible)
    max_dist = min(np.amax(distances), 0.5)
    distances, indices = nbrs.kneighbors(positions.T)
    def likelihood(idx, point):
        if distances[idx][0] >= max_dist:
            alpha = ALPHA_O
        else:
            alpha = ALPHA_I
        return pe.prob_theta_given_lam_stable_set_weight_num(theta, point, TRUE_WEIGHTS[:2], cost, alpha)\
        -pe.prob_theta_given_lam_stable_set_weight_denom(feasible, point, TRUE_WEIGHTS[:2], cost, alpha)
    likelihoods = np.array([likelihood(idx, p) for idx, p in enumerate(positions.T)])
    variance = np.var(likelihoods)
    f = np.reshape(likelihoods.T, xx.shape)
    if second:
        # print np.amin(likelihoods)
        # print "max: " + str(np.amax(likelihoods))
        ax.imshow(np.flip(f, 1).T, cmap='inferno', interpolation='nearest', extent=(-2, 1, -3, 0.5), vmin=-29.8, vmax=-0.9)
        all_vars_dim2.append(variance)
    else:
        # print np.amin(likelihoods)
        # print "max: " + str(np.amax(likelihoods))
        ax.imshow(np.flip(f, 1).T, cmap='inferno', interpolation='nearest', extent=(-3, 1.75, -3, 1.5), vmin=-54.6, vmax=-1.1)
        all_vars_dim1.append(variance)
        xx, yy = np.mgrid[-3:1.75:100j, -3:1.5:100j]
    ax.scatter(feasible[:,0], feasible[:,1], c='C0')
    ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
    if with_belief:
        if second:
            data_means = np.array(dist.particles)[:,2:].T
            kernel = kde(data_means)
            xx, yy = np.mgrid[-2:1:100j, -3:0.5:100j]
        else:
            data_means = np.array(dist.particles)[:,:2].T
            kernel = kde(data_means)
            xx, yy = np.mgrid[-3:1.75:100j, -3:1.5:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)
        cset = ax.contour(xx, yy, f, colors='k')
    ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)
    ax.set_title('variance: %0.2f, size: %d' % (variance, len(feasible)))
def prob_of_truth(dist, ground_truth):
    DOF = len(dist.particles[0])
    cov = np.diag(np.ones(DOF)) * 1
    likelihoods = mvn.pdf(dist.particles, mean=ground_truth, cov=cov)
    return np.sum(likelihoods) / dist.NUM_PARTICLES
def dist_to_truth(dist, ground_truth):
    mean = np.mean(dist.particles, axis=0)
    return np.linalg.norm(mean - ground_truth)
def train_active(dist, data):
    ground_truth_probs = [prob_of_truth(dist, TRUE_MEAN)]
    ground_truth_dists = [dist_to_truth(dist, TRUE_MEAN)]
    data_likelihoods = [-dist.neg_log_likelihood(test_set)]
    for i in range(1, 25):
        func = partial(info_gain, dist)
        pooled = pool.map(func, data)
        print
        expected_infos = [sample[1] for sample in pooled]
        max_idx = np.argmax(expected_infos)
        (theta, feasible) = pooled[max_idx][0]

        dist.weights = dist.reweight_vectorized(theta, feasible)
        dist.resample()
        prob = prob_of_truth(dist, TRUE_WEIGHTS)
        distance = dist_to_truth(dist, TRUE_WEIGHTS)
        ll = dist.neg_log_likelihood(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
    return ground_truth_probs, ground_truth_dists, data_likelihoods
def train_min_cost(dist, data):
    ground_truth_probs = [prob_of_truth(dist, TRUE_MEAN)]
    ground_truth_dists = [dist_to_truth(dist, TRUE_MEAN)]
    data_likelihoods = [-dist.neg_log_likelihood(test_set)]
    for i in range(1, 25):
        func = partial(min_cost, dist)
        pooled = pool.map(func, data)
        print
        expected_costs = [sample[2] for sample in pooled]
        max_idx = np.argmin(expected_costs)
        (theta, feasible) = pooled[max_idx][0]
        dist.weights = dist.reweight_vectorized(theta, feasible)
        dist.resample()
        prob = prob_of_truth(dist, TRUE_WEIGHTS)
        distance = dist_to_truth(dist, TRUE_WEIGHTS)
        ll = dist.neg_log_likelihood(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
    return ground_truth_probs, ground_truth_dists, data_likelihoods
def train_random(dist, data):
    ground_truth_probs = [prob_of_truth(dist, TRUE_MEAN)]
    ground_truth_dists = [dist_to_truth(dist, TRUE_MEAN)]
    data_likelihoods = [-dist.neg_log_likelihood(test_set)]
    for i in range(1, 25):
        theta, feasible = data[i]
        dist.weights = dist.reweight_vectorized(theta, feasible)
        dist.resample()
        prob = prob_of_truth(dist, TRUE_WEIGHTS)
        distance = dist_to_truth(dist, TRUE_WEIGHTS)
        ll = dist.neg_log_likelihood(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
    return ground_truth_probs, ground_truth_dists, data_likelihoods
#########################################################
# f = np.load('../data/sim_rod_training_data.npz')
f = np.load('../data/sim_rod_mug_concat.npz')
idxs = list(range(25))
print idxs
# data_full = f['data_full'][idxs]
all_data = f['data'][idxs]
# poses = f['poses'][idxs]
data = all_data
data_active = data[:8]
test_set  = f['data'][25:35]
for (i, (theta, feasible)) in enumerate(data):
    if len(feasible) > 1000:
        idxs = np.random.choice(len(feasible), size=1000)
        new_feas = feasible[idxs]
        data[i][1] = new_feas

particles = []
weights = []
all_feas = []
for (theta, feasible) in data:
    all_feas.extend(feasible)
all_feas = np.array(all_feas)
mins = np.amin(all_feas, axis=0)
maxes = np.amax(all_feas, axis=0)
ranges = maxes - mins
particles = np.random.uniform(0, 1, size=(NUM_PARTICLES, DOF))
particles *= ranges
particles += mins
weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES
weights = np.array(weights) / np.sum(weights)

dist_active = SetWeightsParticleDistribution(np.copy(particles), np.copy(weights), \
cost, w=TRUE_WEIGHTS, ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)
dist_min_cost = SetWeightsParticleDistribution(np.copy(particles), np.copy(weights), \
cost, w=TRUE_WEIGHTS, ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)
dist_random =SetWeightsParticleDistribution(np.copy(particles), np.copy(weights), \
cost, w=TRUE_WEIGHTS, ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)

initial_prob = prob_of_truth(dist_active, TRUE_WEIGHTS)
initial_dist = dist_to_truth(dist_active, TRUE_WEIGHTS)
initial_ll = -dist_active.neg_log_likelihood(test_set)

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
    probs_random, dists_random, ll_random = train_random(dist_random, data)
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