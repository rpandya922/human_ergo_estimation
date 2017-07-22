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
#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 4
NUM_PARTICLES = 1000
box_size = 0.5
ALPHA_I = 10
ALPHA_O = 5
TRUE_MEAN = np.array([0, 0, 0, 0])
TRUE_WEIGHTS = np.diag([0.5, 0.5, 0.5, 0.5])
def cost(theta, theta_star, w):
    return np.dot(theta - theta_star, np.dot(w, theta - theta_star))
def neg_log_likelihood(data, mean):
    total = 0
    for (theta, feasible) in data:
        total += pe.prob_theta_given_lam_stable_set_weight_num(theta, mean, TRUE_WEIGHTS, cost, 1)
        total -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, mean, TRUE_WEIGHTS, cost, 1)
    return -total
#########################################################
all_data = np.load('./4joint_sim_training_data_mean0.npy')
shuffle(all_data)
test_size = 15
test_data = all_data[:test_size]
data = all_data[test_size:]
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
# particles = np.random.uniform(-3, -1, size=(NUM_PARTICLES, DOF))
dist = SetWeightsParticleDistribution(particles, weights, cost, w=TRUE_WEIGHTS, ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)

fig, axes = plt.subplots(nrows=5, ncols=5)
axes = np.ndarray.flatten(np.array(axes))
fig2, axes2 = plt.subplots(nrows=5, ncols=5)
axes2 = np.ndarray.flatten(np.array(axes2))
ax = axes[0]
ax2 = axes2[0]
# pca = PCA(n_components=2)
# particles_pca = pca.fit_transform(np.array(dist.particles))
# mean_pca = pca.transform([TRUE_MEAN])[0]
# data_means = particles_pca.T
data_means = np.array(dist.particles)[:,:2].T
kernel = kde(data_means)
xx, yy = np.mgrid[-1.75:0.5:100j, -1.25:1.25:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
f = np.reshape(kernel(positions).T, xx.shape)
cfset = ax.contourf(xx, yy, f, cmap='Greens')
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
# ax.scatter(mean_pca[0], mean_pca[1], c='C3', s=200, zorder=2)
ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)

data_means = np.array(dist.particles)[:,2:4].T
kernel = kde(data_means)
xx, yy = np.mgrid[-2:1:100j, -2.5:0.5:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
f = np.reshape(kernel(positions).T, xx.shape)
cfset = ax2.contourf(xx, yy, f, cmap='Greens')
cset = ax2.contour(xx, yy, f, colors='k')
ax2.clabel(cset, inline=1, fontsize=10)
# ax.scatter(mean_pca[0], mean_pca[1], c='C3', s=200, zorder=2)
ax2.scatter(TRUE_MEAN[2], TRUE_MEAN[3], c='C3', s=200, zorder=2)
print neg_log_likelihood(test_data, TRUE_MEAN)
ll = dist.neg_log_likelihood(test_data)
ax.set_title(ll)
ax2.set_title(ll)
plt.pause(0.2)
for i in range(1, 25):
    theta, feasible = data[i]
    dist.weights = dist.reweight(theta, feasible)
    dist.resample()

    ax = axes[i]
    ax2 = axes2[i]
    # particles_pca = pca.fit_transform(np.array(dist.particles))
    # mean_pca, theta_pca = pca.transform([TRUE_MEAN, theta])
    # feasible_pca = pca.transform(feasible)
    # data_means = particles_pca.T
    data_means = np.array(dist.particles)[:,:2].T
    kernel = kde(data_means)
    xx, yy = np.mgrid[-1.75:0.5:100j, -1.25:1.25:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='Greens')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    # ax.scatter(feasible_pca[:,0], feasible_pca[:,1], c='C0')
    # ax.scatter(theta_pca[0], theta_pca[1], c='C2', s=200, zorder=2)
    # ax.scatter(mean_pca[0], mean_pca[1], c='C3', s=200, zorder=2)
    ax.scatter(feasible[:,0], feasible[:,1], c='C0')
    ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
    ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)

    data_means = np.array(dist.particles)[:,2:4].T
    kernel = kde(data_means)
    xx, yy = np.mgrid[-2:1:100j, -2.5:0.5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax2.contourf(xx, yy, f, cmap='Greens')
    cset = ax2.contour(xx, yy, f, colors='k')
    ax2.clabel(cset, inline=1, fontsize=10)
    # ax.scatter(mean_pca[0], mean_pca[1], c='C3', s=200, zorder=2)
    ax2.scatter(feasible[:,2], feasible[:,3], c='C0')
    ax2.scatter(theta[2], theta[3], c='C2', s=200, zorder=2)
    ax2.scatter(TRUE_MEAN[2], TRUE_MEAN[3], c='C3', s=200, zorder=2)

    ll = dist.neg_log_likelihood(test_data)
    ax.set_title(ll)
    ax2.set_title(ll)
    plt.pause(0.2)
plt.show()
