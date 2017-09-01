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
from tqdm import tqdm
from scipy.stats import multivariate_normal as mvn

parser = argparse.ArgumentParser()
parser.add_argument('--sets', nargs='+', type=int, default=list(range(10)))
parser.add_argument('--weights', nargs='+', type=int, default=list(range(5)))
args = parser.parse_args()
############################################################
# CONSTANTS/FUNCTIONS
DOF = 7
ALPHA = 40
ALPHA_I = 1
ALPHA_O = 1
FEASIBLE_SIZE = 1000
TRUE_MEAN = np.array([0, 0, 0, 0, 0, 0, 0])
TRUE_WEIGHTS = np.array([[4, 3, 2, 1, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5, 6], \
                         [1, 5, 3, 5, 7, 7, 4], [8, 4, 6, 1, 3, 6, 8]])
TRUE_WEIGHTS = TRUE_WEIGHTS / np.linalg.norm(TRUE_WEIGHTS, axis=1).reshape(-1, 1)
TEST_SET_SIZE = 300
NUM_PARTICLES = 1000
NUM_TRAIN_ITERATIONS = 10
training_data_size = 500
DISTRIBUTION_DATA_FOLDER = '../data/exp_likelihood_handpicked_new'
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
    idx = np.argmax(probs)
    return (feasible[idx], feasible)
def preprocess_feasible(data, poses, get_feasible=True):
    new_data_full = []
    new_poses = []
    for i in tqdm(range(data.shape[1])):
    # for i in range(70):
        if get_feasible:
            feasible = np.array(get_feasible_set(data, i))
        else:
            feasible = np.array(data[i])
        try:
            if feasible == None:
                continue
        except:
            pass
        if len(feasible) <= 2:
            continue
        try:
            weights = 1 / kde(feasible.T).evaluate(feasible.T)
        except:
            continue
        weights /= np.sum(weights)
        uniform_feasible_full = feasible[np.random.choice(len(feasible), p=weights,\
        size=min(1000, len(feasible)))]
        new_data_full.append(uniform_feasible_full)
        new_poses.append(poses[i])
    return new_data_full, new_poses
def train_active(dist, data, ground_truth):
    # data = full_data[np.random.choice(len(full_data), size=8)]
    all_particles = [np.copy(dist.particles)]
    ground_truth_probs = [utils.prob_of_truth(dist, ground_truth)]
    ground_truth_dists = [utils.dist_to_truth(dist, ground_truth)]
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

        prob = utils.prob_of_truth(dist, ground_truth)
        distance = utils.dist_to_truth(dist, ground_truth)
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
    # data = full_data[np.random.choice(len(full_data), size=8)]
    all_particles = [np.copy(dist.particles)]
    ground_truth_probs = [utils.prob_of_truth(dist, ground_truth)]
    ground_truth_dists = [utils.dist_to_truth(dist, ground_truth)]
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

        prob = utils.prob_of_truth(dist, ground_truth)
        distance = utils.dist_to_truth(dist, ground_truth)
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
    # data = full_data[np.random.choice(len(full_data), size=8)]
    all_particles = [np.copy(dist.particles)]
    ground_truth_probs = [utils.prob_of_truth(dist, ground_truth)]
    ground_truth_dists = [utils.dist_to_truth(dist, ground_truth)]
    data_likelihoods = [-dist.neg_log_likelihood_mean(test_set)]
    for i in range(1, NUM_TRAIN_ITERATIONS):
        print "\rRandom on iteration %d of 9" % i,
        sys.stdout.flush()
        idx = np.random.choice(len(data))
        (theta, feasible) = data[idx]
        dist.weights = dist.reweight(theta, feasible)
        dist.resample()

        prob = utils.prob_of_truth(dist, ground_truth)
        distance = utils.dist_to_truth(dist, ground_truth)
        ll = dist.neg_log_likelihood_mean(test_set)
        ground_truth_probs.append(prob)
        ground_truth_dists.append(distance)
        data_likelihoods.append(-ll)
        all_particles.append(np.copy(dist.particles))
        plt.pause(0.01)
    print
    return ground_truth_probs, ground_truth_dists, data_likelihoods, all_particles
def get_test_sets():
    datasets = []
    for weights_idx in args.weights:
        print "Preprocessing test set %d..." % weights_idx
        weights = TRUE_WEIGHTS[weights_idx]
        # data, poses = np.array(preprocess_feasible(np.load('../data/sim_data_rod.npy'), np.load('../data/rod_full_cases.npz')['pose_samples']))
        data, poses = np.array(preprocess_feasible(np.load('../data/rod_and_mug_data.npz')['data'], \
        np.load('../data/rod_and_mug_data.npz')['poses'], False))
        training_data = []
        for i in range(len(data)):
            feasible = data[i]
            probs = get_distribution(feasible, cost, weights, ALPHA)
            training_data.append(create_sample(feasible, probs))
        datasets.append(training_data)
    return datasets
##############################################################
datasets = []
for weights_idx in args.weights:
    print "Preprocessing weight %d..." % weights_idx
    weights = TRUE_WEIGHTS[weights_idx]
    # data, poses = np.array(preprocess_feasible(np.load('../data/rod_handpicked_data.npy'), np.load('../data/rod_handpicked_cases.npz')['pose_samples']))
    # data, poses = np.array(preprocess_feasible(np.load('../data/sim_data_rod.npy'), np.load('../data/rod_full_cases.npz')['pose_samples']))
    data, poses = np.array(preprocess_feasible(np.load('../data/rod_and_mug_data.npz')['data'], \
    np.load('../data/rod_and_mug_data.npz')['poses'], False))
    training_data = []
    for i in range(len(data)):
        feasible = data[i]
        probs = get_distribution(feasible, cost, weights, ALPHA)
        training_data.append(create_sample(feasible, probs))
    datasets.append(training_data)
# good_idxs = [2, 5, 18]
# bad_idxs = [0, 1, 11, 14, 23]
feasible_idxs = [2, 5, 18, 0, 1, 11, 14, 23]
test_sets = get_test_sets()
objects = np.load('../data/rod_and_mug_data.npz')['objects']
def info_gain(dist, x):
    return (x, dist.info_gain(x[1], num_boxes=20), dist.expected_cost(x[1]))
def min_cost(dist, x):
    return (x, 0, dist.expected_cost(x[1]))
if __name__ == '__main__':
    pool = mp.Pool(8)

    for weight_idx in args.weights:
        ground_truth_weights = TRUE_WEIGHTS[weight_idx]
        all_data = datasets[weight_idx]
        test_set = test_sets[weight_idx][:TEST_SET_SIZE]
        for set_idx in args.sets:
            np.random.seed(set_idx + (1000*weight_idx))
            # test_set = all_data[:TEST_SET_SIZE]
            data = all_data[TEST_SET_SIZE:]
            idxs = np.random.choice(len(data), size=8)
            # idxs = feasible_idxs
            data = np.array(data)[idxs]
            chosen_poses = poses[TEST_SET_SIZE:][idxs]
            chosen_objects = objects[TEST_SET_SIZE:][idxs]
            # test_set = all_data[25:35]
            # data = all_data[:8]
            # data = np.array(all_data)[TEST_SET_SIZE:]
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
            dist_passive = SetMeanParticleDistribution(np.copy(particles), np.copy(weights), utils.cost, m=TRUE_MEAN, \
            ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=0.01)
            dist_random = SetMeanParticleDistribution(np.copy(particles), np.copy(weights), utils.cost, m=TRUE_MEAN, \
            ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=0.01)

            initial_prob = utils.prob_of_truth(dist_active, ground_truth_weights)
            initial_dist = utils.dist_to_truth(dist_active, ground_truth_weights)
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
                            'expected_costs': expected_costs, 'training_poses': chosen_poses[:], \
                            'particles_active': particles_active, 'particles_passive': particles_passive, \
                            'particles_random': particles_random, 'training_objects': chosen_objects[:]}
            output = open('%s/set%s_param%s.pkl' % (DISTRIBUTION_DATA_FOLDER, set_idx, weight_idx), 'wb')
            pickle.dump(pickle_dict, output)
            output.close()