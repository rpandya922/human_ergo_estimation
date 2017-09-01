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
from distribution import SetWeightsParticleDistribution
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
parser.add_argument('--means', nargs='+', type=int, default=list(range(5)))
args = parser.parse_args()
################################################################################
# CONSTANTS/FUNCTIONS
DOF = 7
ALPHA = 40
ALPHA_I = 1
ALPHA_O = 1
FEASIBLE_SIZE = 1000
TRUE_WEIGHTS = np.array([1, 1, 1, 1, 1, 1, 1])
TRUE_MEANS = np.array([[-2.60, -0.82, 0.42, 0.46, 1.43, -2.91, -0.12],
                       [2.21, -0.05, 1.83, -2.49, -1.44, 1.83, 1.66],
                       [-1.76, -2.36, -1.11, -2.36, -2.78, -0.88, 0.94],
                       [1.67, 0.63, -2.34, -0.73, 1.27, -0.38, 1.67],
                       [-0.48, -0.45, 1.78, 2.97, 1.49, 2.65, -2.23]])
TRUE_MEANS = TRUE_MEANS / np.linalg.norm(TRUE_WEIGHTS, axis=1).reshape(-1, 1)
TEST_SET_SIZE = 300
NUM_PARTICLES = 1000
NUM_TRAIN_ITERATIONS = 10
training_data_size = 500
DISTRIBUTION_DATA_FOLDER = '../data/mean_rod_only'
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
    nums = np.array([pe.prob_theta_given_lam_stable_set_weight_num(theta, ground_truth, TRUE_WEIGHTS, cost, ALPHA) for theta in feasible])
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feasible, ground_truth, TRUE_WEIGHTS, cost, ALPHA)
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
            d = SetWeightsParticleDistribution(dist.particles, dist.weights, dist.cost, dist.w, dist.ALPHA_I, dist.ALPHA_O, dist.h)
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
    # data = full_data[np.random.choice(len(full_data), size=8)]
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
    # data = full_data[np.random.choice(len(full_data), size=8)]
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
    datasets = []
    for mean_idx in args.means:
        print "Preprocessing test set %d..." % mean_idx
        mean = TRUE_MEANS[mean_idx]
        data, poses = np.array(preprocess_feasible(np.load('../data/sim_data_rod.npy'), np.load('../data/rod_full_cases.npz')['pose_samples']))
        # data, poses = np.array(preprocess_feasible(np.load('../data/rod_and_mug_data.npz')['data'], \
        # np.load('../data/rod_and_mug_data.npz')['poses'], False))
        training_data = []
        for i in range(len(data)):
            feasible = data[i]
            probs = get_distribution(feasible, cost, mean, ALPHA)
            training_data.append(create_sample(feasible, probs))
        datasets.append(training_data)
    return datasets
#########################################################
datasets = []
for mean_idx in args.means:
    print "Preprocessing mean %d..." % mean_idx
    mean = TRUE_MEAN[mean_idx]
    data, poses = np.array(preprocess_feasible(np.load('../data/sim_data_rod.npy'), np.load('../data/rod_full_cases.npz')['pose_samples']))
    # data, poses = np.array(preprocess_feasible(np.load('../data/rod_and_mug_data.npz')['data'], \
    # np.load('../data/rod_and_mug_data.npz')['poses'], False))
    training_data = []
    for i in range(len(data)):
        feasible = data[i]
        probs = get_distribution(feasible, cost, mean, ALPHA)
        training_data.append(create_sample(feasible, probs))
    datasets.append(training_data)
test_sets = get_test_sets()
def info_gain(dist, x):
    return (x, dist.info_gain(x[1], num_boxes=20), dist.expected_cost(x[1]))
def min_cost(dist, x):
    return (x, 0, dist.expected_cost(x[1]))
if __name__ == '__main__':
    pool = mp.Pool(8)

    for mean_idx in args.means:
        ground_truth_mean = TRUE_MEANS[mean_idx]
        all_data = datasets[mean_idx]
        test_set = test_sets[mean_idx][:TEST_SET_SIZE]
        for set_idx in args.sets:
            np.random.seed(set_idx + (1000 * mean_idx))
            data = all_data[TEST_SET_SIZE:]
            idxs = np.random.choice(len(data), size=8)
            chosen_poses = poses[TEST_SET_SIZE:][idxs]

            particles = []
            weights = []
            while len(particles) < NUM_PARTICLES:
                p = np.random.randn(DOF, 1).T[0]
                p = p / np.linalg.norm(p, axis=0)
                if p[0] >= 0 and p[1] >= 0 and p[2] >= 0 and p[3] >= 0 and p[4] >= 0 and p[5] >= 0 and p[6] >= 0:
                    particles.append(p)
            particles = np.array(particles)
            weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

            dist_active = SetWeightsParticleDistribution(np.copy(particles), np.copy(weights), cost, w=TRUE_WEIGHTS,\
            ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)
            dist_passive = SetWeightsParticleDistribution(np.copy(particles), np.copy(weights), cost, w=TRUE_WEIGHTS,\
            ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)
            dist_random = SetWeightsParticleDistribution(np.copy(particles), np.copy(weights), cost, w=TRUE_WEIGHTS,\
            ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)

            initial_prob = prob_of_truth(dist_active, ground_truth_mean)
            initial_dist = dist_to_truth(dist_active, ground_truth_mean)
            initial_ll = -dist_active.neg_log_likelihood_mean(test_set)

            probs_active, dists_active, ll_active, expected_infos, actual_infos, particles_active = train_active(dist_active, data, ground_truth_mean)
            probs_passive, dists_passive, ll_passive, expected_costs, particles_passive = train_min_cost(dist_passive, data, ground_truth_mean)
            probs_random, dists_random, ll_random, particles_random = train_random(dist_random, data, ground_truth_mean)

            pickle_dict = {'training_data': data, 'mean': ground_truth_mean, \
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
                            'particles_random': particles_random}
            output = open('%s/set%s_param%s.pkl' % (DISTRIBUTION_DATA_FOLDER, set_idx, weight_idx), 'wb')
            pickle.dump(pickle_dict, output)
            output.close()