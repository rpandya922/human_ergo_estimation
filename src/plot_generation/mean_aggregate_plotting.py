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
import readline
import argparse
import pickle
import csv
from tqdm import tqdm
################################################################################
# CONSTANTS/FUNCTIONS
DATA_LOCATION = '../data/mean_rod_only'
TRUE_WEIGHTS = np.ones(7)
def calc_error_bounds(data):
    n = np.array(data).shape[1]
    sample_mean = np.mean(data, axis=0)
    sample_std = np.std(data, axis=0)
    return (1 / np.sqrt(n)* sample_std)
def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)
def prob_of_truth(dist, ground_truth):
    DOF = len(dist.particles[0])
    cov = np.diag(np.ones(DOF)) * 0.25
    likelihoods = mvn.pdf(dist.particles, mean=ground_truth, cov=cov)
    return np.sum(likelihoods) / dist.NUM_PARTICLES
def dist_to_truth(dist, ground_truth):
    mode = dist.distribution_mode()
    return np.linalg.norm(mode - ground_truth)
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
# def plot_poses(data, poses):
#     env, human, robot, target, target_desc = load_environment_file('../data/rod_full_problem_def.npz')
#     env.SetViewer('qtcoin')
#     newrobots = []
#     for ind in range(15):
#         newrobot = RaveCreateRobot(env,human.GetXMLId())
#         newrobot.Clone(human,0)
#         for link in newrobot.GetLinks():
#             for geom in link.GetGeometries():
#                 geom.SetTransparency(0.8)
#         newrobots.append(newrobot)
#     for link in robot.GetLinks():
#         for geom in link.GetGeometries():
#             geom.SetTransparency(0.8)
#     for (i, (theta, feasible)) in enumerate(data):
#         target.SetTransform(poses[i])
#         with env:
#             inds = np.array(np.linspace(0,len(feasible)-1,15),int)
#             for j,ind in enumerate(inds):
#                 newrobot = newrobots[j]
#                 env.Add(newrobot,True)
#                 newrobot.SetTransform(human.GetTransform())
#                 newrobot.SetDOFValues(feasible[ind], human.GetActiveManipulator().GetArmIndices())
#         env.UpdatePublishedBodies()
        # raw_input('Displaying pose ' + str(i) + ', press <Enter> to continue:')
def plot_belief(ax, particles, ground_truth, second=False):
    data_means = particles.T
    kernel = kde(data_means)
    xx, yy = np.mgrid[-2:1:100j, -3:0.5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='Greens')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.scatter(ground_truth[0], ground_truth[1], c='C3', s=200, zorder=2)
def plot_beliefs(all_particles):
    fig, axes = plt.subplots(nrows=5, ncols=2)
    axes = np.ndarray.flatten(np.array(axes))
def neg_log_likelihood(data, ground_truth):
    total = 0
    for (theta, feasible) in data:
        total += pe.prob_theta_given_lam_stable_set_weight_num(theta, ground_truth, TRUE_WEIGHTS, cost, 1)
        total -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, ground_truth, TRUE_WEIGHTS, cost, 1)
    return -total
def calc_test_set_likelihood(all_particles, test_set):
    likelihoods = []
    for particles in tqdm(all_particles):
        weights = np.ones(particles.shape[0]) / particles.shape[0]
        dist = SetWeightsParticleDistribution(np.array(particles), np.array(weights), \
        cost, TRUE_WEIGHTS, 1, 1, 0.03)
        likelihoods.append(-dist.neg_log_likelihood_mean(test_set))
    return likelihoods
def calc_prob_of_truth(all_particles, ground_truth):
    probs = []
    for particles in all_particles:
        weights = np.ones(particles.shape[0]) / particles.shape[0]
        dist = SetWeightsParticleDistribution(np.array(particles), np.array(weights), \
        cost, TRUE_WEIGHTS, 1, 1, 0.03)
        probs.append(prob_of_truth(dist, ground_truth))
    return probs
def calc_dist_to_truth(all_particles, ground_truth):
    dists = []
    for particles in all_particles:
        weights = np.ones(particles.shape[0]) / particles.shape[0]
        dist = SetWeightsParticleDistribution(np.array(particles), np.array(weights), \
        cost, TRUE_WEIGHTS, 1, 1, 0.03)
        dists.append(dist_to_truth(dist, ground_truth))
    return dists
def expected_cost(feasible, ground_truth):
    probs = pe.prob_theta_given_lam_stable_set_weight_num(feasible, ground_truth, TRUE_WEIGHTS, cost, 1)
    probs -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, ground_truth, TRUE_WEIGHTS, cost, 1)
    probs = np.exp(probs)
    costs = cost(feasible, ground_truth, TRUE_WEIGHTS)
    return np.sum(probs * costs)
def value_of_info_metric(all_particles, test_set, ground_truth):
    true_feas_costs = [(feas, expected_cost(feas, ground_truth)) for theta, feas in test_set]
    true_min_cost_feas, true_min_cost = min(true_feas_costs, key=lambda x: x[1])
    vals = []
    for particles in tqdm(all_particles):
        weights = np.ones(particles.shape[0]) / particles.shape[0]
        dist = SetWeightsParticleDistribution(np.array(particles), np.array(weights), \
        cost, TRUE_WEIGHTS, 1, 1, 0.03)
        # mode = dist.distribution_mode()
        mode = np.mean(particles, axis=0)
        feas_costs = [(feas, dist.expected_cost2(feas, mode)) for theta, feas in test_set]
        min_cost_feas, min_cost = min(feas_costs, key=lambda x: x[1])
        vals.append(expected_cost(min_cost_feas, ground_truth))
    return vals, [true_min_cost] * len(all_particles)
################################################################################
value_infos_ground_truth = []
log_likelihoods_ground_truth = []

ground_truth_probs_active = []
ground_truth_dists_active = []
data_likelihoods_active = []
value_infos_active = []

ground_truth_probs_passive = []
ground_truth_dists_passive = []
data_likelihoods_passive = []
value_infos_passive = []

ground_truth_probs_random = []
ground_truth_dists_random = []
data_likelihoods_random = []
value_infos_random = []
test_set = []
weights = []
for set_idx in tqdm(range(10)):
    for param_idx in tqdm(range(5)):
        try:
            pkl_file = open('%s/set%s_param%s.pkl' % (DATA_LOCATION, set_idx, param_idx), 'rb')
        except:
            continue
        data = pickle.load(pkl_file)
        test_set = data['test_set']
        mean = data['mean']
        # plot_info_bars(data['expected_infos'], data['actual_infos'])
        # plot_cost_bars(data['expected_costs'])
        # plot_final_belief(data['distribution_active'], weights, 'active')
        # plot_final_belief(data['distribution_passive'], weights, 'passive')
        # plot_final_belief(data['distribution_random'], weights, 'random')

        # initial_prob = data['initial_prob']
        # initial_dist = data['initial_dist']
        # initial_ll = data['initial_ll']
        #
        # probs_active = data['probs_active']
        # probs_passive = data['probs_passive']
        # probs_random = data['probs_random']
        #
        # dists_active = data['distances_active']
        # dists_passive = data['distances_passive']
        # dists_random = data['distances_random']
        #
        # ll_active = data['ll_active']
        # ll_passive = data['ll_passive']
        # ll_random = data['ll_random']

        probs_active = calc_prob_of_truth(data['particles_active'], mean)
        probs_passive = calc_prob_of_truth(data['particles_passive'], mean)
        probs_random = calc_prob_of_truth(data['particles_random'], mean)

        dists_active = calc_dist_to_truth(data['particles_active'], mean)
        dists_passive = calc_dist_to_truth(data['particles_passive'], mean)
        dists_random = calc_dist_to_truth(data['particles_random'], mean)

        ll_active = calc_test_set_likelihood(data['particles_active'], test_set)
        ll_passive = calc_test_set_likelihood(data['particles_passive'], test_set)
        ll_random = calc_test_set_likelihood(data['particles_random'], test_set)

        val_info_active, val_info_gt = value_of_info_metric(data['particles_active'], test_set, mean)
        val_info_passive, _ = value_of_info_metric(data['particles_passive'], test_set, mean)
        val_info_random, _ = value_of_info_metric(data['particles_random'], test_set, mean)

        # probs_active[0] = initial_prob
        # probs_passive[0] = initial_prob
        # probs_random[0] = initial_prob
        ground_truth_probs_active.append(probs_active)
        ground_truth_probs_passive.append(probs_passive)
        ground_truth_probs_random.append(probs_random)

        # dists_active[0] = initial_dist
        # dists_passive[0] = initial_dist
        # dists_random[0] = initial_dist
        ground_truth_dists_active.append(dists_active)
        ground_truth_dists_passive.append(dists_passive)
        ground_truth_dists_random.append(dists_random)

        # ll_active[0] = initial_ll
        # ll_passive[0] = initial_ll
        # ll_random[0] = initial_ll
        data_likelihoods_active.append(ll_active)
        data_likelihoods_passive.append(ll_passive)
        data_likelihoods_random.append(ll_random)

        value_infos_active.append(val_info_active)
        value_infos_passive.append(val_info_passive)
        value_infos_random.append(val_info_random)
        value_infos_ground_truth.append(val_info_gt)

        log_likelihoods_ground_truth.append([-neg_log_likelihood(test_set, mean)] * 10)

# plot_poses(data['training_data'], data['training_poses'])
# plot_final_belief(data['distribution_active'], weights, 'active')
# plot_final_belief(data['distribution_passive'], weights, 'passive')
# plot_final_belief(data['distribution_random'], weights, 'random')
x = list(range(10))

values_active = np.mean(value_infos_active, axis=0)
values_passive = np.mean(value_infos_passive, axis=0)
values_random = np.mean(value_infos_random, axis=0)
values_gt = np.mean(value_infos_ground_truth, axis=0)
error_values_active = calc_error_bounds(value_infos_active)
error_values_passive = calc_error_bounds(value_infos_passive)
error_values_random = calc_error_bounds(value_infos_random)
error_values_gt = calc_error_bounds(value_infos_ground_truth)

fig = plt.figure()
ax = fig.add_subplot(221)
ax.set_xlim(0, 9)
ax.plot()
ax.plot(values_random, c='C0', label='randomly selected')
ax.plot(values_passive, c='C1', label='passive learning')
ax.plot(values_active, c='C2', label='active learning')
ax.plot(values_gt, c='C3', label='ground truth')
ax.fill_between(x, values_random - error_values_random, values_random + error_values_random, color='C0', alpha=0.3)
ax.fill_between(x, values_passive - error_values_passive, values_passive + error_values_passive, color='C1', alpha=0.3)
ax.fill_between(x, values_active - error_values_active, values_active + error_values_active, color='C2', alpha=0.3)
ax.fill_between(x, values_gt - error_values_gt, values_gt + error_values_gt, color='C3', alpha=0.3)
ax.legend(loc='upper left')
ax.set_xlabel('iteration')
ax.set_ylabel('value of information (?)')
plt.pause(0.1)

probs_random = np.mean(ground_truth_probs_random, axis=0)
probs_passive = np.mean(ground_truth_probs_passive, axis=0)
probs_active = np.mean(ground_truth_probs_active, axis=0)
error_probs_random = calc_error_bounds(ground_truth_probs_random)
error_probs_passive = calc_error_bounds(ground_truth_probs_passive)
error_probs_active = calc_error_bounds(ground_truth_probs_active)

ax = fig.add_subplot(222)
ax.set_xlim(0, 9)
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

ax = fig.add_subplot(223)
ax.set_xlim(0, 9)
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


error_ll_random = calc_error_bounds(data_likelihoods_random)
error_ll_passive = calc_error_bounds(data_likelihoods_passive)
error_ll_active = calc_error_bounds(data_likelihoods_active)
error_ll_gt = calc_error_bounds(log_likelihoods_ground_truth)
ll_random = np.mean(data_likelihoods_random, axis=0)
ll_passive = np.mean(data_likelihoods_passive, axis=0)
ll_active = np.mean(data_likelihoods_active, axis=0)
ll_gt = np.mean(log_likelihoods_ground_truth, axis=0)

ax = fig.add_subplot(224)
ax.set_xlim(0, 9)
ax.plot(ll_random, label='randomly selected')
ax.plot(ll_passive, label='passive learning')
ax.plot(ll_active, label='active learning')
ax.plot(ll_gt, label='ground truth')
ax.fill_between(x, ll_random - error_ll_random, ll_random + error_ll_random, alpha=0.3)
ax.fill_between(x, ll_passive - error_ll_passive, ll_passive + error_ll_passive, alpha=0.3)
ax.fill_between(x, ll_active - error_ll_active, ll_active + error_ll_active, alpha=0.3)
ax.fill_between(x, ll_gt - error_ll_gt, ll_gt + error_ll_gt, alpha=0.3)
ax.legend(loc='upper left')
ax.set_xlabel('iteration')
ax.set_ylabel('log likelihood of test set')
plt.pause(0.1)
plt.show()