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
from scipy.stats import multivariate_normal as mvn
############################################################
# CONSTANTS/FUNCTIONS
DOF = 7
ALPHA = 40
ALPHA_I = 1
ALPHA_O = 1
FEASIBLE_SIZE = 1000
TRUE_MEAN = np.array([0, 0, 0, 0, 0, 0, 0])
TRUE_WEIGHTS = np.array([[4, 3, 2, 1, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1]])
TRUE_WEIGHTS = TRUE_WEIGHTS / np.linalg.norm(TRUE_WEIGHTS, axis=1).reshape(-1, 1)
TEST_SET_SIZE = 5
NUM_PARTICLES = 2000
NUM_TRAIN_ITERATIONS = 10
training_data_size = 500

def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)
def get_feasible_set(data, pose):
    sets_list = data[:,pose]
    feasible = []
    for s in sets_list:
        feasible.extend(s)
    feasible = np.array(feasible)
    if len(feasible) == 0:
        return None
    return feasible
def get_distribution(feasible, cost, ground_truth, ALPHA):
    nums = np.array([pe.prob_theta_given_lam_stable_set_weight_num(theta, TRUE_MEAN, ground_truth, cost, ALPHA) for theta in feasible])
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feasible, TRUE_MEAN, ground_truth, cost, ALPHA)
    return np.exp(nums - denom)
def create_sample(feasible, probs):
    # idx = np.random.choice(len(feasible), p=probs)
    idx = np.argmax(probs)
    return (feasible[idx], feasible)
def preprocess_feasible(data, poses):
    new_data_full = []
    new_poses = []
    for i in range(data.shape[1]):
        feasible = np.array(get_feasible_set(data, i))
        try:
            if feasible == None:
                continue
        except:
            pass
        if len(feasible) <= 6:
            continue
        # print len(feasible)
        try:
            weights = 1 / kde(feasible.T).evaluate(feasible.T)
        except:
            continue
        weights /= np.sum(weights)
        uniform_feasible_full = feasible[np.random.choice(len(feasible), p=weights,\
        size=min(1000, len(feasible)))]
        new_data_full.append(uniform_feasible_full)
        new_poses.append(poses[i])
        print "\r%d" % i,
        sys.stdout.flush()
    print
    return new_data_full, new_poses
def train_active(dist, data, ground_truth):
    ground_truth_probs = [utils.prob_of_truth(dist, ground_truth)]
    ground_truth_dists = [utils.dist_to_truth(dist, ground_truth)]
    data_likelihoods = [-dist.neg_log_likelihood_mean(test_set)]
    for i in range(1, NUM_TRAIN_ITERATIONS):
        print "\rActive on iteration %d of 9" % i,
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

        prob = utils.prob_of_truth(dist, ground_truth)
        distance = utils.dist_to_truth(dist, ground_truth)
        ll = dist.neg_log_likelihood_mean(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
        plt.pause(0.01)
    print
    return ground_truth_probs, ground_truth_dists, data_likelihoods
def train_min_cost(dist, data, ground_truth):
    ground_truth_probs = [utils.prob_of_truth(dist, ground_truth)]
    ground_truth_dists = [utils.dist_to_truth(dist, ground_truth)]
    data_likelihoods = [-dist.neg_log_likelihood_mean(test_set)]
    for i in range(1, NUM_TRAIN_ITERATIONS):
        print "\rMin cost on iteration %d of 9" % i,
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

        prob = utils.prob_of_truth(dist, ground_truth)
        distance = utils.dist_to_truth(dist, ground_truth)
        ll = dist.neg_log_likelihood_mean(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
        plt.pause(0.01)
    print
    return ground_truth_probs, ground_truth_dists, data_likelihoods
def train_random(dist, data, ground_truth):
    ground_truth_probs = [utils.prob_of_truth(dist, ground_truth)]
    ground_truth_dists = [utils.dist_to_truth(dist, ground_truth)]
    data_likelihoods = [-dist.neg_log_likelihood_mean(test_set)]
    for i in range(1, NUM_TRAIN_ITERATIONS):
        print "\rRandom on iteration %d of 9" % i,
        sys.stdout.flush()
        (theta, feasible) = data[i % 8]
        dist.weights = dist.reweight(theta, feasible)
        dist.resample()

        prob = utils.prob_of_truth(dist, ground_truth)
        distance = utils.dist_to_truth(dist, ground_truth)
        ll = dist.neg_log_likelihood_mean(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
        plt.pause(0.01)
    print
    return ground_truth_probs, ground_truth_dists, data_likelihoods
##############################################################
datasets = []
for weights in TRUE_WEIGHTS:
    data, poses = np.array(preprocess_feasible(np.load('../data/sim_data_rod.npy'), np.load('../data/rod_full_cases.npz')['pose_samples']))
    print "preprocessed"
    training_data = []
    for i in range(len(data)):
        feasible = data[i]
        probs = get_distribution(feasible, cost, weights, ALPHA)
        training_data.append(create_sample(feasible, probs))
    datasets.append(training_data)

def info_gain(dist, x):
    return (x, dist.info_gain(x[1], num_boxes=20), dist.expected_cost(x[1]))
def min_cost(dist, x):
    return (x, 0, dist.expected_cost(x[1]))
if __name__ == '__main__':
    pool = mp.Pool(8)

    ground_truth_probs_active = []
    ground_truth_dists_active = []
    data_likelihoods_active = []

    ground_truth_probs_passive = []
    ground_truth_dists_passive = []
    data_likelihoods_passive = []

    ground_truth_probs_random = []
    ground_truth_dists_random = []
    data_likelihoods_random = []
    for weight_idx, all_data in enumerate(datasets):
        ground_truth_weights = TRUE_WEIGHTS[weight_idx]
        for _ in range(2):
            test_set = all_data[:TEST_SET_SIZE]
            data = all_data[TEST_SET_SIZE:]
            idxs = idxs = np.random.choice(len(data), size=8)
            data = np.array(data)[idxs]

            particles = []
            weights = []
            while len(particles) < NUM_PARTICLES:
                p = np.random.randn(DOF, 1).T[0]
                p = p / np.linalg.norm(p, axis=0)
                if p[0] >= 0 and p[1] >= 0 and p[2] >= 0 and p[3] >= 0 and p[4] >= 0 and p[5] >= 0 and p[6] >= 0:
                    particles.append(p)
            particles = np.array(particles)
            weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

            dist_active = SetMeanParticleDistribution(particles, weights, utils.cost, m=TRUE_MEAN, \
            ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=0.01)
            dist_min_cost = SetMeanParticleDistribution(np.copy(particles), np.copy(weights), utils.cost, m=TRUE_MEAN, \
            ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=0.01)
            dist_random = SetMeanParticleDistribution(np.copy(particles), np.copy(weights), utils.cost, m=TRUE_MEAN, \
            ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=0.01)

            initial_prob = utils.prob_of_truth(dist_active, ground_truth_weights)
            initial_dist = utils.dist_to_truth(dist_active, ground_truth_weights)
            initial_ll = -dist_active.neg_log_likelihood_mean(test_set)

            probs_active, dists_active, ll_active = train_active(dist_active, data, ground_truth_weights)
            print "finished active"
            probs_min_cost, dists_min_cost, ll_min_cost = train_min_cost(dist_min_cost, data, ground_truth_weights)
            print "finished passive"
            probs_random, dists_random, ll_random = train_random(dist_random, data, ground_truth_weights)
            print "finished random"

            probs_active[0] = initial_prob
            probs_min_cost[0] = initial_prob
            probs_random[0] = initial_prob
            ground_truth_probs_active.append(probs_active)
            ground_truth_probs_passive.append(probs_min_cost)
            ground_truth_probs_random.append(probs_random)

            dists_active[0] = initial_dist
            dists_min_cost[0] = initial_dist
            dists_random[0] = initial_dist
            ground_truth_dists_active.append(dists_active)
            ground_truth_dists_passive.append(dists_min_cost)
            ground_truth_dists_random.append(dists_random)

            ll_active[0] = initial_ll
            ll_min_cost[0] = initial_ll
            ll_random[0] = initial_ll
            data_likelihoods_active.append(ll_active)
            data_likelihoods_passive.append(ll_min_cost)
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