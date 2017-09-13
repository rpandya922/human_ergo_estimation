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
two_pi = 2 * np.pi
DEFAULT_DATA_FILE = 'handlebars_sim_data.npz'
DEFAULT_PROBLEM_DEF_FILE = 'handlebars_problem_def.npz'
def plot_feas(ax, ax2, theta, feasible):
    # ax.set_xlim(-3, 1.75)
    # ax.set_ylim(-3, 1.5)
    # ax2.set_xlim(-2, 1)
    # ax2.set_ylim(-3, 0.5)

    ax.scatter(feasible[:,0], feasible[:,1], c='C0')
    ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
    ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)

    ax2.scatter(feasible[:,2], feasible[:,3], c='C0')
    ax2.scatter(theta[2], theta[3], c='C2', s=200, zorder=2)
    ax2.scatter(TRUE_MEAN[2], TRUE_MEAN[3], c='C3', s=200, zorder=2)
    plt.pause(0.01)
def get_likelihood(dist, test_set):
    true_likelihoods = 0
    likelihood = dist.neg_log_likelihood(test_set)
    for (theta, feasible) in test_set:
        ll = pe.prob_theta_given_lam_stable_set_weight_num(theta, dist.m, TRUE_WEIGHTS, dist.cost, 1)
        ll -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, dist.m, TRUE_WEIGHTS, dist.cost, 1)
        true_likelihoods += -ll
    return likelihood, true_likelihoods
#########################################################
datafile = utils.prefilled_input('Simulation data file: ', DEFAULT_DATA_FILE)
problem_def_file = utils.prefilled_input('Problem definition file: ', DEFAULT_PROBLEM_DEF_FILE)

f = np.load('../data/' + datafile)
# idxs = list(range(50))
data_full = f['data_full']#[idxs]
all_data = f['data']#[idxs]
test_set = f['data']#[np.random.choice(len(f['data']), size=10)]
poses = f['poses']#[idxs]
env, human, robot, target, target_desc = utils.load_environment_file('../data/' + problem_def_file)
env.SetViewer('qtcoin')
data = all_data
# fig = plt.figure()
# ax = fig.add_subplot(111)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
#
# lh_fig = plt.figure()
# lh_ax = lh_fig.add_subplot(111)
# lh_fig2 = plt.figure()
# lh_ax2 = lh_fig2.add_subplot(111)

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
    # ax.cla()
    # ax2.cla()
    # lh_ax.cla()
    # lh_ax2.cla()
    # ax.set_title("Feasible sets to choose from (dim 1&2)")
    # ax2.set_title("Feasible sets to choose from (dim 3&4)")
    # lh_ax.set_title("Likelihood (dim 1&2)")
    # lh_ax2.set_title("Likelihood (dim 3&4)")
    # plot_feas(ax, ax2, theta, feasible)
    # utils.plot_likelihood_heatmap_norm_weights(lh_ax, theta[:2], feasible[:,:2], TRUE_WEIGHTS1, vmin=-8.1, vmax=-3)
    # utils.plot_likelihood_heatmap_norm_weights(lh_ax2, theta[2:], feasible[:,2:], TRUE_WEIGHTS2, vmin=-8.1, vmax=-3)
    target.SetTransform(poses[i])
    feas_full = data_full[i]
    with env:
        inds = np.array(np.linspace(0,len(feas_full)-1,15),int)
        for j,ind in enumerate(inds):
            newrobot = newrobots[j]
            env.Add(newrobot,True)
            newrobot.SetTransform(human.GetTransform())
            newrobot.SetDOFValues(feas_full[ind], human.GetActiveManipulator().GetArmIndices())
    env.UpdatePublishedBodies()
    plt.pause(0.01)
    raw_input('Displaying pose ' + str(i) + ', press <Enter> to continue:')