from __future__ import division
import sys
sys.path.insert(0, '../')
from openravepy import *
import prpy
import numpy as np
import seaborn
from matplotlib import rc
# rc('text', usetex=True)
# rc('font', **{'family':'sans-serif', 'serif': 'Times', 'sans-serif':['Helvetica']})
seaborn.set_style('ticks')
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
ACTIVE_COLOR = '#ffaa00'
# PASSIVE_COLOR = '#24e2cc'
PASSIVE_COLOR = '#46c6db'
# RANDOM_COLOR = '#704300'
RANDOM_COLOR = '#c68400'
DATA_LOCATION = '../data/user_study'
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
    cov = np.diag(np.ones(DOF)) * 1
    likelihoods = mvn.pdf(dist.particles, mean=ground_truth, cov=cov)
    return np.sum(likelihoods) / dist.NUM_PARTICLES
def dist_to_truth(dist, ground_truth):
    # mode = dist.distribution_mode()
    mode = np.mean(dist.particles, axis=0)
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
def load_object_desc(desc_string):
    result = eval(desc_string)
    for tsr_name in result['human_tsrs'].keys():
        tsr_obj = prpy.tsr.TSR(**result['human_tsrs'][tsr_name])
        result['human_tsrs'][tsr_name] = tsr_obj
    for tsr_name in result['robot_tsrs'].keys():
        tsr_obj = prpy.tsr.TSR(**result['robot_tsrs'][tsr_name])
        result['robot_tsrs'][tsr_name] = tsr_obj
    return result
def load(filename):
    return np.load(filename)
def load_txt(filename):
    with open(filename, 'r') as file_handle:
        object_string = file_handle.read()
    return load_object_desc(object_string)
def load_environment_file(filename):
    problem_def = load(filename)
    human_file = problem_def['human_file'].tostring()
    robot_file = problem_def['robot_file'].tostring()
    object_file = problem_def['object_file'].tostring()
    target_desc = load_txt('../data/' + object_file)
    human_base_pose = problem_def['human_base_pose']
    robot_base_pose = problem_def['robot_base_pose']
    object_start_pose = problem_def['object_start_pose']
    problem_def.close()
    return load_environment(human_file, robot_file, object_file,
            human_base_pose, robot_base_pose, object_start_pose)
def load_environment(human_file, robot_file, object_file,
        human_base_pose, robot_base_pose, object_start_pose):
    env = Environment()

    #Add the human
    human = env.ReadKinBodyXMLFile(human_file)
    env.AddKinBody(human)

    env.GetCollisionChecker().SetCollisionOptions(0)
    manip = human.SetActiveManipulator('rightarm')
    human.SetTransform(human_base_pose)

    human.GetLink('Hips').SetVisible(False)
    hand_joints = []
    hand_joints.append(human.GetJointIndex('JLFing11'))
    hand_joints.append(human.GetJointIndex('JLFing21'))
    hand_joints.append(human.GetJointIndex('JLFing31'))
    hand_joints.append(human.GetJointIndex('JLFing41'))
    hand_joints.append(human.GetJointIndex('JLFing10'))
    hand_joints.append(human.GetJointIndex('JLFing20'))
    hand_joints.append(human.GetJointIndex('JLFing30'))
    hand_joints.append(human.GetJointIndex('JLFing40'))
    hand_joints.append(human.GetJointIndex('JRFing11'))
    hand_joints.append(human.GetJointIndex('JRFing21'))
    hand_joints.append(human.GetJointIndex('JRFing31'))
    hand_joints.append(human.GetJointIndex('JRFing41'))
    hand_joints.append(human.GetJointIndex('JRFing10'))
    hand_joints.append(human.GetJointIndex('JRFing20'))
    hand_joints.append(human.GetJointIndex('JRFing30'))
    hand_joints.append(human.GetJointIndex('JRFing40'))
    human.SetDOFValues([0.5]*16, hand_joints)

    #Add the robot
    with env:
         robot = env.ReadRobotXMLFile(robot_file)
    #Add the object
    target_desc = load_txt('../data/' + object_file)
    with env:
        target = env.ReadKinBodyXMLFile(target_desc['object_file'])
        print target_desc['object_file']
        env.AddKinBody(target)
        target.SetTransform(object_start_pose)

    return env, human, robot, target, target_desc
def plot_poses(data, poses):
    env, human, robot, target, target_desc = load_environment_file('../data/handlebars_problem_def.npz')
    env.SetViewer('qtcoin')
    newrobots = []
    for ind in range(15):
        newrobot = RaveCreateRobot(env,human.GetXMLId())
        newrobot.Clone(human,0)
        for link in newrobot.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(0.8)
        newrobots.append(newrobot)
    for link in robot.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(0.8)
    for (i, (theta, feasible)) in enumerate(data):
        target.SetTransform(poses[i])
        with env:
            inds = np.array(np.linspace(0,len(feasible)-1,15),int)
            for j,ind in enumerate(inds):
                newrobot = newrobots[j]
                env.Add(newrobot,True)
                newrobot.SetTransform(human.GetTransform())
                newrobot.SetDOFValues(feasible[ind], human.GetActiveManipulator().GetArmIndices())
        env.UpdatePublishedBodies()
        raw_input('Displaying pose ' + str(i) + ', press <Enter> to continue:')
def plot_belief(ax, particles, ground_truth):
    data_means = particles.T
    kernel = kde(data_means)
    xx, yy = np.mgrid[-3.14:3.14:100j, -3.14:3.14:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='Greens')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
def plot_beliefs(all_particles, ground_truth, title=''):
    fig, axes = plt.subplots(nrows=3, ncols=2)
    axes = np.ndarray.flatten(np.array(axes))
    fig.suptitle(title)
    for i, particles, in enumerate(all_particles):
        plot_belief(axes[i], particles[:,:2], ground_truth)
        plt.pause(0.01)
def neg_log_likelihood(data, ground_truth):
    total = 0
    for (theta, feasible) in data:
        total += pe.prob_theta_given_lam_stable_set_weight_num(theta, ground_truth, TRUE_WEIGHTS, cost, 1)
        total -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, ground_truth, TRUE_WEIGHTS, cost, 1)
    return -total
def calc_test_set_likelihood(all_particles, test_set):
    likelihoods = []
    for particles in all_particles:
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
    for particles in all_particles:
        weights = np.ones(particles.shape[0]) / particles.shape[0]
        dist = SetWeightsParticleDistribution(np.array(particles), np.array(weights), \
        cost, TRUE_WEIGHTS, 1, 1, 0.03)
        # mode = dist.distribution_mode()
        mode = np.mean(particles, axis=0)
        feas_costs = [(feas, dist.expected_cost2(feas, mode)) for theta, feas in test_set]
        min_cost_feas, min_cost = min(feas_costs, key=lambda x: x[1])
        vals.append(expected_cost(min_cost_feas, ground_truth))
    return vals, [true_min_cost] * len(all_particles)
def entropy_metric(all_particles):
    ents = []
    for particles in all_particles:
        weights = np.ones(particles.shape[0]) / particles.shape[0]
        dist = SetWeightsParticleDistribution(np.array(particles), np.array(weights), \
        cost, TRUE_WEIGHTS, 1, 1, 0.03)
        ents.append(dist.entropy(num_boxes=10))
    return ents
def argmax_distance(all_particles_active, all_particles_passive):
    distances = []
    for i in range(len(all_particles_active)):

        weights = np.ones(1000) / 1000
        dist_active = SetWeightsParticleDistribution(np.copy(particles_active), np.copy(weights), cost, w=TRUE_WEIGHTS,\
        ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=RESAMPLING_VARIANCE)
        dist_passive = SetWeightsParticleDistribution(np.copy(particles), np.copy(weights), cost, w=TRUE_WEIGHTS,\
        ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=RESAMPLING_VARIANCE)
def print_final_mode(all_particles):
    particles = all_particles[len(all_particles) - 1]
    weights = np.ones(1000) / 1000
    dist = SetWeightsParticleDistribution(particles, weights, cost, w=TRUE_WEIGHTS,\
    ALPHA_I=1, ALPHA_O=1, h=0.1)
    print dist.distribution_mode()
################################################################################
value_infos_ground_truth = []
log_likelihoods_ground_truth = []

data_likelihoods_active = []
value_infos_active = []
entropys_active = []
normalized_likelihoods_active = []

data_likelihoods_passive = []
value_infos_passive = []
entropys_passive = []
normalized_likelihoods_passive = []

data_likelihoods_random = []
value_infos_random = []
entropys_random = []
normalized_likelihoods_random = []
# in bike simulation: folder data/mean_bike_sim_overnight, set 9: active
# does well, set 14: passive does poorly
test_set = []
weights = []
# for set_idx in tqdm([9]):
for param_idx in range(1, 6):
    try:
        pkl_file = open('%s/subject%s_beleifs.pkl' % (DATA_LOCATION, param_idx), 'rb')
    except:
        continue
    data = pickle.load(pkl_file)
    # test_set = data['test_set']
    test_set = np.load('../data/user_study/subject%s_test_set.npz' % param_idx)['test_set']
    # test_set = np.load('../data/user_study/subject%s_training_set.npz' % param_idx)['training_set']
    # mean = data['mean']
    mean = np.zeros(7)
    print_final_mode(data['particles_passive'])
    # plot_poses(data['training_data'], data['training_poses'])
    # plot_beliefs(data['particles_active'], mean, 'active')
    # plot_beliefs(data['particles_passive'], mean, 'passive')
    # plot_beliefs(data['particles_random'], mean, 'random')
    # plot_info_bars(data['expected_infos'], data['actual_infos'])
    # plot_cost_bars(data['expected_costs'])
    # # plot_final_belief(data['distribution_active'], weights, 'active')
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

    ll_active = calc_test_set_likelihood(data['particles_active'], test_set)
    ll_passive = calc_test_set_likelihood(data['particles_passive'], test_set)
    # ll_random = calc_test_set_likelihood(data['particles_random'], test_set)

    val_info_active, val_info_gt = value_of_info_metric(data['particles_active'], test_set, mean)
    val_info_passive, _ = value_of_info_metric(data['particles_passive'], test_set, mean)
    # val_info_random, _ = value_of_info_metric(data['particles_random'], test_set, mean)

    entropy_active = entropy_metric(data['particles_active'])
    entropy_passive = entropy_metric(data['particles_passive'])
    # entropy_random = entropy_metric(data['particles_random'])

    # ll_active[0] = initial_ll
    # ll_passive[0] = initial_ll
    # ll_random[0] = initial_ll
    data_likelihoods_active.append(ll_active)
    data_likelihoods_passive.append(ll_passive)
    # data_likelihoods_random.append(ll_random)

    value_infos_active.append(val_info_active)
    value_infos_passive.append(val_info_passive)
    # value_infos_random.append(val_info_random)
    value_infos_ground_truth.append(val_info_gt)

    entropys_active.append(entropy_active)
    entropys_passive.append(entropy_passive)
    # entropys_random.append(entropy_random)

    ground_truth_likelihood = np.array([-neg_log_likelihood(test_set, mean)] * 6)
    log_likelihoods_ground_truth.append(ground_truth_likelihood)

    normalized_likelihoods_active.append(np.exp(ll_active - ground_truth_likelihood))
    normalized_likelihoods_passive.append(np.exp(ll_passive - ground_truth_likelihood))
    # normalized_likelihoods_random.append(np.exp(ll_random - ground_truth_likelihood))
plt.show()

# plot_poses(data['training_data'], data['training_poses'])
# plot_final_belief(data['distribution_active'], weights, 'active')
# plot_final_belief(data['distribution_passive'], weights, 'passive')
# plot_final_belief(data['distribution_random'], weights, 'random')
x = list(range(6))

values_active = np.mean(value_infos_active, axis=0)
values_passive = np.mean(value_infos_passive, axis=0)
# values_random = np.mean(value_infos_random, axis=0)
values_gt = np.mean(value_infos_ground_truth, axis=0)
error_values_active = calc_error_bounds(value_infos_active)
error_values_passive = calc_error_bounds(value_infos_passive)
# error_values_random = calc_error_bounds(value_infos_random)
error_values_gt = calc_error_bounds(value_infos_ground_truth)

fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(221)
ax.set_xlim(0, 5)
ax.plot()
# ax.plot(values_random, c=RANDOM_COLOR, lw=3, label='randomly selected')
ax.plot(values_passive, c=PASSIVE_COLOR, lw=3, label='passive learning')
ax.plot(values_active, c=ACTIVE_COLOR, lw=3, label='active learning')
# ax.plot(values_gt, c='k', linestyle='dashed', lw=3, label='ground truth')
# ax.fill_between(x, values_random - error_values_random, values_random + error_values_random, color=RANDOM_COLOR, alpha=0.3)
ax.fill_between(x, values_passive - error_values_passive, values_passive + error_values_passive, color=PASSIVE_COLOR, alpha=0.3)
ax.fill_between(x, values_active - error_values_active, values_active + error_values_active, color=ACTIVE_COLOR, alpha=0.3)
# ax.fill_between(x, values_gt - error_values_gt, values_gt + error_values_gt, color='k', alpha=0.3)
# lgnd = ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left")
# lgnd.legendHandles[0]._sizes = [30]
# lgnd.legendHandles[1]._sizes = [30]
# lgnd.legendHandles[2]._sizes = [30]
ax.set_xlabel('iteration')
ax.set_ylabel('value of information')
plt.pause(0.1)

# error_ll_random = calc_error_bounds(data_likelihoods_random)
error_ll_passive = calc_error_bounds(data_likelihoods_passive)
error_ll_active = calc_error_bounds(data_likelihoods_active)
error_ll_gt = calc_error_bounds(log_likelihoods_ground_truth)
# ll_random = np.mean(data_likelihoods_random, axis=0)
ll_passive = np.mean(data_likelihoods_passive, axis=0)
ll_active = np.mean(data_likelihoods_active, axis=0)
ll_gt = np.mean(log_likelihoods_ground_truth, axis=0)

ax = fig.add_subplot(222)
ax.set_xlim(0, 5)
# ax.plot(ll_random, c=RANDOM_COLOR, lw=3, label='randomly selected')
ax.plot(ll_passive, c=PASSIVE_COLOR, lw=3, label='passive learning')
ax.plot(ll_active, c=ACTIVE_COLOR, lw=3, label='active learning')
# ax.plot(ll_gt, c='k', linestyle='dashed', lw=3, label='ground truth')
# ax.fill_between(x, ll_random - error_ll_random, ll_random + error_ll_random, color=RANDOM_COLOR, alpha=0.3)
ax.fill_between(x, ll_passive - error_ll_passive, ll_passive + error_ll_passive, color=PASSIVE_COLOR, alpha=0.3)
ax.fill_between(x, ll_active - error_ll_active, ll_active + error_ll_active, color=ACTIVE_COLOR, alpha=0.3)
# ax.fill_between(x, ll_gt - error_ll_gt, ll_gt + error_ll_gt, color='k',  alpha=0.3)
ax.set_xlabel('iteration')
ax.set_ylabel('log likelihood of test set')
plt.pause(0.1)

# ents_random = np.mean(entropys_random, axis=0)
ents_passive = np.mean(entropys_passive, axis=0)
ents_active = np.mean(entropys_active, axis=0)
# error_ents_random = calc_error_bounds(entropys_random)
error_ents_passive = calc_error_bounds(entropys_passive)
error_ents_active = calc_error_bounds(entropys_active)

ax = fig.add_subplot(223)
ax.set_xlim(0, 5)
# ax.plot(ents_random, c=RANDOM_COLOR, lw=3, label='randomly selected')
ax.plot(ents_passive, c=PASSIVE_COLOR, lw=3, label='passive learning')
ax.plot(ents_active, c=ACTIVE_COLOR, lw=3, label='active learning')
# ax.fill_between(x, ents_random - error_ents_random, ents_random + error_ents_random, color=RANDOM_COLOR, alpha=0.3)
ax.fill_between(x, ents_passive - error_ents_passive, ents_passive + error_ents_passive, color=PASSIVE_COLOR, alpha=0.3)
ax.fill_between(x, ents_active - error_ents_active, ents_active + error_ents_active, color=ACTIVE_COLOR, alpha=0.3)
ax.set_xlabel('iteration')
ax.set_ylabel('entropy of belief')
plt.pause(0.1)

# norm_ll_random = np.mean(normalized_likelihoods_random, axis=0)
norm_ll_passive = np.mean(normalized_likelihoods_passive, axis=0)
norm_ll_active = np.mean(normalized_likelihoods_active, axis=0)
# error_norm_ll_random = calc_error_bounds(normalized_likelihoods_random)
error_norm_ll_passive = calc_error_bounds(normalized_likelihoods_passive)
error_norm_ll_active = calc_error_bounds(normalized_likelihoods_active)

ax = fig.add_subplot(224)
ax.set_xlim(0, 5)
# ax.plot(norm_ll_random, c=RANDOM_COLOR, lw=3, label='randomly selected')
ax.plot(norm_ll_passive, c=PASSIVE_COLOR, lw=3, label='passive learning')
ax.plot(norm_ll_active, c=ACTIVE_COLOR, lw=3, label='active learning')
# ax.plot([1] * 10, c='k', linestyle='dashed', lw=3, label='ground truth')
# ax.fill_between(x, norm_ll_random - error_norm_ll_random, norm_ll_random + error_norm_ll_random, color=RANDOM_COLOR, alpha=0.3)
ax.fill_between(x, norm_ll_passive - error_norm_ll_passive, norm_ll_passive + error_norm_ll_passive, color=PASSIVE_COLOR, alpha=0.3)
ax.fill_between(x, norm_ll_active - error_norm_ll_active, norm_ll_active + error_norm_ll_active, color=ACTIVE_COLOR, alpha=0.3)
ax.set_xlabel('iteration')
ax.set_ylabel('normalized likelihood')
plt.pause(0.1)
plt.show()