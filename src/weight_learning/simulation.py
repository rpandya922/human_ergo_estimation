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
from distribution import SetMeanParticleDistribution
import multiprocessing as mp
from functools import partial
from random import shuffle
from sklearn.neighbors import NearestNeighbors
import utils
import readline
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
NUM_PARTICLES = 1000
two_pi = 2 * np.pi
DEFAULT_DATA_FILE = 'sim_rod_weight_learning.npz'
DEFAULT_PROBLEM_DEF_FILE = 'rod_full_problem_def.npz'
def plot_feas(data):
    fig, axes = plt.subplots(nrows=5, ncols=5)
    axes = np.ndarray.flatten(np.array(axes))
    fig2, axes2 = plt.subplots(nrows=5, ncols=5)
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
def prob_of_truth(dist, ground_truth):
    data_means = dist.particles.T
    kernel = kde(data_means)
    return kernel(ground_truth)
#########################################################
datafile = utils.prefilled_input('Simulation data file: ', DEFAULT_DATA_FILE)
problem_def_file = utils.prefilled_input('Problem definition file: ', DEFAULT_PROBLEM_DEF_FILE)

f = np.load('../data/' + datafile)
idxs = list(range(25))
print idxs
data_full = f['data_full'][idxs]
all_data = f['data'][idxs]
poses = f['poses'][idxs]
env, human, robot, target, target_desc = utils.load_environment_file('../data/' + problem_def_file)
env.SetViewer('qtcoin')
data = all_data
fig, axes = plt.subplots(nrows=5, ncols=5)
axes = np.ndarray.flatten(np.array(axes))
fig2, axes2 = plt.subplots(nrows=5, ncols=5)
axes2 = np.ndarray.flatten(np.array(axes2))

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
plot_feas(data)
for (i, (theta, feasible)) in enumerate(data):
    utils.plot_likelihood_heatmap_norm_weights(axes[i], theta[:2], feasible[:,:2], TRUE_WEIGHTS1, vmin=-7.5, vmax=-2)
    utils.plot_likelihood_heatmap_norm_weights(axes2[i], theta[2:], feasible[:,2:], TRUE_WEIGHTS2, vmin=-7.5, vmax=-2)
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
    # raw_input('Displaying pose ' + str(i) + ', press <Enter> to continue:')
fig.suptitle('dim 1&2 Particles: ' + str(NUM_PARTICLES) + ' alpha_i: ' + str(ALPHA_I) +\
             ' alpha_o: ' + str(ALPHA_O))
fig2.suptitle('dim 3&4 Particles: ' + str(NUM_PARTICLES) + ' alpha_i: ' + str(ALPHA_I) +\
             ' alpha_o: ' + str(ALPHA_O))
for (i, (theta, feasible)) in enumerate(data):
    if len(feasible) > 1000:
        idxs = np.random.choice(len(feasible), size=1000)
        new_feas = feasible[idxs]
        data[i][1] = new_feas
particles = []
weights = []
while len(particles) < NUM_PARTICLES:
    p = np.random.randn(DOF, 1).T[0]
    p = p / np.linalg.norm(p, axis=0)
    if p[0] >= 0 and p[1] >= 0 and p[2] >= 0 and p[3] >= 0:
        particles.append(p)
particles = np.array(particles)
weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES
dist = SetMeanParticleDistribution(particles, weights, utils.cost, m=TRUE_MEAN, \
ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=0.01)
fig, axes = plt.subplots(nrows=5, ncols=5)
axes = np.ndarray.flatten(np.array(axes))
fig2, axes2 = plt.subplots(nrows=5, ncols=5)
axes2 = np.ndarray.flatten(np.array(axes2))
fig.suptitle('dim 1&2 Particles: ' + str(NUM_PARTICLES) + ' alpha_i: ' + str(ALPHA_I) +\
             ' alpha_o: ' + str(ALPHA_O))
fig2.suptitle('dim 3&4 Particles: ' + str(NUM_PARTICLES) + ' alpha_i: ' + str(ALPHA_I) +\
             ' alpha_o: ' + str(ALPHA_O))
particles1 = dist.particles[:,:2] / np.linalg.norm(dist.particles[:,:2], axis=1).reshape(-1, 1)
particles2 = dist.particles[:,2:] / np.linalg.norm(dist.particles[:,2:], axis=1).reshape(-1, 1)
utils.plot_belief(axes[0], particles1, TRUE_WEIGHTS1)
utils.plot_belief(axes2[0], particles2, TRUE_WEIGHTS2)
plt.pause(0.1)
1/0
for i in range(1, len(axes)):
    theta, feasible = data[i % len(data)]
    dist.weights = dist.reweight(theta, feasible)
    dist.resample()
    particles1 = dist.particles[:,:2] / np.linalg.norm(dist.particles[:,:2], axis=1).reshape(-1, 1)
    particles2 = dist.particles[:,2:] / np.linalg.norm(dist.particles[:,2:], axis=1).reshape(-1, 1)
    utils.plot_belief(axes[i], particles1, TRUE_WEIGHTS1)
    utils.plot_belief(axes2[i], particles2, TRUE_WEIGHTS2)
    axes[i].set_title(prob_of_truth(dist, TRUE_WEIGHTS))
    axes2[i].set_title(prob_of_truth(dist, TRUE_WEIGHTS))
    plt.pause(0.1)
plt.show()