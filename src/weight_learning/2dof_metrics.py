from __future__ import division
import sys
sys.path.insert(0, '../')
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
import readline
import argparse
import pickle
from tqdm import tqdm
from scipy.stats import multivariate_normal as mvn

parser = argparse.ArgumentParser()
parser.add_argument('--sets', nargs='+', type=int, default=list(range(10)))
parser.add_argument('--weights', nargs='+', type=int, default=list(range(5)))
args = parser.parse_args()
################################################################################
# CONSTANTS/FUNCTIONS
DOF = 2
l1 = 3
l2 = 3
two_pi = 2 * np.pi
ALPHA = 40
ALPHA_I = 1
ALPHA_O = 1
FEASIBLE_SIZE = 1000
TRUE_MEAN = np.array([1, 2])
TRUE_WEIGHTS = np.array([[2.60, 0.82],
                       [2.21, 0.05],
                       [1.76, 2.36],
                       [1.67, 0.63],
                       [0.48, 0.45]])
TRUE_WEIGHTS = TRUE_WEIGHTS / np.linalg.norm(TRUE_WEIGHTS, axis=1).reshape(-1, 1)
TEST_SET_SIZE = 300
NUM_PARTICLES = 500
NUM_TRAIN_ITERATIONS = 10
TRAINING_DATA_SIZE = 500
DISTRIBUTION_DATA_FOLDER = '../data/2dof_arm_weights_testing'
def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)
def create_box(upper_left, lower_right, box_size=0.5):
    feas = []
    x = upper_left[:]
    while x[1] >= lower_right[1]:
        while x[0] <= lower_right[0]:
            feas.append(x[:])
            x[0] += box_size
        x[0] = upper_left[0]
        x[1] -= box_size
    return np.unique(feas, axis=0)
def create_ellipse(x0, y0, a, b):
    rand = np.random.uniform(0, 1, size=(1000, 2)) * np.array([2*a, 2*b])
    rand += np.array([x0 - a, y0 - b])
    feas = []
    vals = np.sum(((rand - np.array([x0, y0])) / np.array([a, b])) ** 2, axis=1)
    for i, v in enumerate(vals):
        if v <= 1:
            feas.append(rand[i])
    return np.array(feas)
def get_theta(x, y):
    inv_cos = np.arccos( ((l1**2) + (l2**2) - (x**2) - (y**2)) / (2 * l1 * l2))
    theta_prime = np.arcsin(l2 * np.sin(inv_cos) / np.sqrt((x**2) + (y**2)))
    theta1_partial = np.arctan2(x, y)
    theta2 = np.pi - inv_cos
    if np.isnan(inv_cos) or np.isnan(theta_prime):
        return []
    thetas = [[theta1_partial + theta_prime, -theta2], [theta1_partial - theta_prime, theta2]]
    # thetas = [[theta1_partial - theta_prime, theta2]]
    if thetas[0][0] > np.pi:
        thetas[0][0] -= two_pi
    elif thetas[0][0] < -np.pi:
        thetas[0][0] += two_pi
    if thetas[1][0] > np.pi:
        thetas[1][0] -= two_pi
    elif thetas[1][0] < -np.pi:
        thetas[1][0] += two_pi
    return thetas
def create_sample(feas, ground_truth):
    nums = pe.prob_theta_given_lam_stable_set_weight_num(feas, TRUE_MEAN, ground_truth, cost, ALPHA)
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feas, TRUE_MEAN, ground_truth, cost, ALPHA)
    probs = np.exp(nums - denom)
    chosen_idx = np.argmax(probs)
    chosen = feas[chosen_idx]
    return (chosen, feas)
def create_sample_from_xy(obj, ground_truth):
    feas = []
    for (x, y) in obj:
        feas.extend(get_theta(x, y))
    feas = np.array(feas)
    nums = pe.prob_theta_given_lam_stable_set_weight_num(feas, TRUE_MEAN, ground_truth, cost, ALPHA)
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feas, TRUE_MEAN, ground_truth, cost, ALPHA)
    probs = np.exp(nums - denom)
    chosen_idx = np.argmax(probs)
    chosen = feas[chosen_idx]
    return (chosen, feas)
def get_distribution(feasible, cost, ground_truth, ALPHA):
    nums = np.array([pe.prob_theta_given_lam_stable_set_weight_num(theta, ground_truth, TRUE_WEIGHTS, cost, ALPHA) for theta in feasible])
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feasible, ground_truth, TRUE_WEIGHTS, cost, ALPHA)
    return np.exp(nums - denom)
def prob_of_truth(dist, ground_truth):
    DOF = len(dist.particles[0])
    cov = np.diag(np.ones(DOF)) * 0.0625
    likelihoods = mvn.pdf(dist.particles, mean=ground_truth, cov=cov)
    return np.sum(likelihoods) / dist.NUM_PARTICLES
def dist_to_truth(dist, ground_truth):
    mode = dist.distribution_mode()
    return np.linalg.norm(mode - ground_truth)
def train_active(dist, data, ground_truth):
    all_particles = [np.copy(dist.particles)]
    ground_truth_probs = [prob_of_truth(dist, ground_truth)]
    ground_truth_dists = [dist_to_truth(dist, ground_truth)]
    data_likelihoods = [-dist.neg_log_likelihood_mean(test_set)]
    all_expected_infos = []
    all_actual_infos = []
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

        prob = prob_of_truth(dist, ground_truth)
        distance = dist_to_truth(dist, ground_truth)
        ll = dist.neg_log_likelihood_mean(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
        all_expected_infos.append(expected_infos[:])
        all_actual_infos.append(actual_infos[:])
        all_particles.append(np.copy(dist.particles))
        plt.pause(0.01)
    print
    return ground_truth_probs, ground_truth_dists, data_likelihoods, all_expected_infos, all_actual_infos, all_particles
def train_min_cost(dist, data, ground_truth):
    all_particles = [np.copy(dist.particles)]
    ground_truth_probs = [prob_of_truth(dist, ground_truth)]
    ground_truth_dists = [dist_to_truth(dist, ground_truth)]
    data_likelihoods = [-dist.neg_log_likelihood_mean(test_set)]
    all_expected_costs = []
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

        prob = prob_of_truth(dist, ground_truth)
        distance = dist_to_truth(dist, ground_truth)
        ll = dist.neg_log_likelihood_mean(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
        all_expected_costs.append(expected_costs[:])
        all_particles.append(np.copy(dist.particles))
        plt.pause(0.01)
    print
    return ground_truth_probs, ground_truth_dists, data_likelihoods, all_expected_costs, all_particles
def train_random(dist, data, ground_truth):
    all_particles = [np.copy(dist.particles)]
    ground_truth_probs = [prob_of_truth(dist, ground_truth)]
    ground_truth_dists = [dist_to_truth(dist, ground_truth)]
    data_likelihoods = [-dist.neg_log_likelihood_mean(test_set)]
    for i in range(1, NUM_TRAIN_ITERATIONS):
        print "\rRandom on iteration %d of 9" % i,
        sys.stdout.flush()
        idx = np.random.choice(len(data))
        (theta, feasible) = data[idx]
        dist.weights = dist.reweight(theta, feasible)
        dist.resample()

        prob = prob_of_truth(dist, ground_truth)
        distance = dist_to_truth(dist, ground_truth)
        ll = dist.neg_log_likelihood_mean(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
        all_particles.append(np.copy(dist.particles))
        plt.pause(0.01)
    print
    return ground_truth_probs, ground_truth_dists, data_likelihoods, all_particles
def get_test_sets():
    test_set = []
    np.random.seed(0)
    # for i in range(TEST_SET_SIZE):
    #     feasible = create_ellipse((np.random.uniform() * 4) - 2, (np.random.uniform() * 4) - 2, \
    #     np.random.uniform() * 3, np.random.uniform() * 3)
    #     test_set.append(feasible)
    for i in range(TEST_SET_SIZE):
        obj = np.random.uniform()
        if obj <= 1:
            x = np.random.randint(-6, 3)
            y = np.random.randint(-3, 6)
            feasible = create_box([x, y], [x + 3, y - 3])
        else:
            print "line"
            y = np.random.randint(-5, 5)
            feasible = create_box([-6, y], [6, y])
        test_set.append(feasible)
    datasets = []
    for weight_idx in list(range(5)):
        print "Preprocessing test set %d..." % weight_idx
        weights = TRUE_WEIGHTS[weight_idx]
        testing_data = []
        for test_feas in test_set:
            testing_data.append(create_sample_from_xy(test_feas, weights))
        datasets.append(testing_data)
    return datasets
#########################################################
training_set = []
np.random.seed(1)
for i in range(TRAINING_DATA_SIZE):
    feasible = create_ellipse((np.random.uniform() * 4) - 2, (np.random.uniform() * 4) - 2, \
    np.random.uniform() * 3, np.random.uniform() * 3)
    training_set.append(feasible)
datasets = []
for weight_idx in list(range(5)):
    print "Preprocessing weight %d..." % weight_idx
    weights = TRUE_WEIGHTS[weight_idx]
    training_data = []
    for test_feas in training_set:
        training_data.append(create_sample_from_xy(test_feas, weights))
    datasets.append(training_data)

test_sets = get_test_sets()
def info_gain(dist, x):
    return (x, dist.info_gain(x[1], num_boxes=20))
def min_cost(dist, x):
    return (x, 0, dist.expected_cost2(x[1], dist.distribution_mode()))
if __name__ == '__main__':
    pool = mp.Pool(8)

    for weight_idx in args.weights:
        ground_truth_weights = TRUE_WEIGHTS[weight_idx]
        all_data = datasets[weight_idx]
        test_set = test_sets[weight_idx][:]
        for set_idx in args.sets:
            np.random.seed(set_idx + (1000 * weight_idx))
            data = all_data[:]
            idxs = np.random.choice(len(data), size=8)
            data = np.array(data)[idxs]

            particles = []
            weights = []
            while len(particles) < NUM_PARTICLES:
                p = np.random.randn(DOF, 1).T[0]
                p = p / np.linalg.norm(p, axis=0)
                if p[0] >= 0 and p[1] >= 0:
                    particles.append(p)
            particles = np.array(particles)
            weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

            dist_active = SetMeanParticleDistribution(np.copy(particles), np.copy(weights), cost, m=TRUE_MEAN,\
            ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)
            dist_passive = SetMeanParticleDistribution(np.copy(particles), np.copy(weights), cost, m=TRUE_MEAN,\
            ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)
            dist_random = SetMeanParticleDistribution(np.copy(particles), np.copy(weights), cost, m=TRUE_MEAN,\
            ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)

            initial_prob = prob_of_truth(dist_active, ground_truth_weights)
            initial_dist = dist_to_truth(dist_active, ground_truth_weights)
            initial_ll = -dist_active.neg_log_likelihood_mean(test_set)

            probs_active, dists_active, ll_active, expected_infos, actual_infos, particles_active = train_active(dist_active, data, ground_truth_weights)
            probs_passive, dists_passive, ll_passive, expected_costs, particles_passive = train_min_cost(dist_passive, data, ground_truth_weights)
            probs_random, dists_random, ll_random, particles_random = train_random(dist_random, data, ground_truth_weights)

            pickle_dict = {'training_data': data, 'weights': ground_truth_weights, \
                            'distribution_active': dist_active, 'distribution_passive': dist_passive, \
                            'distribution_random': dist_random, 'probs_active': probs_active, \
                            'probs_passive': probs_passive, 'probs_random': probs_random, \
                            'distances_active': dists_active, 'distances_passive': dists_passive, \
                            'distances_random': dists_random, 'll_active': ll_active, \
                            'll_passive': ll_passive, 'll_random': ll_random, \
                            'initial_prob': initial_prob, 'initial_dist': initial_dist, \
                            'initial_ll': initial_ll, 'test_set': test_set, \
                            'expected_infos': expected_infos, 'actual_infos': actual_infos, \
                            'expected_costs': expected_costs, 'particles_random': particles_random, \
                            'particles_active': particles_active, 'particles_passive': particles_passive}
            output = open('%s/set%s_param%s.pkl' % (DISTRIBUTION_DATA_FOLDER, set_idx, weight_idx), 'wb')
            pickle.dump(pickle_dict, output)
            output.close()