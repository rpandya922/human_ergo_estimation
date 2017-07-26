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
NUM_PARTICLES = 1001
box_size = 0.5
ALPHA_I = 5
ALPHA_O = 2.5
TRUE_MEAN = np.array([0, 0, 0, 0])
TRUE_WEIGHTS = np.array([1, 1, 1, 1])
def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)
def neg_log_likelihood(data, mean):
    total = 0
    for (theta, feasible) in data:
        total += pe.prob_theta_given_lam_stable_set_weight_num(theta, mean, TRUE_WEIGHTS, cost, 1)
        total -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, mean, TRUE_WEIGHTS, cost, 1)
    return -total
def plot_updates(all_data):
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
    # ll = dist.neg_log_likelihood(test_data)
    # ax.set_title(ll)
    # ax2.set_title(ll)
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

        # ll = dist.neg_log_likelihood(test_data)
        # ax.set_title(ll)
        # ax2.set_title(ll)
        plt.pause(0.2)
    plt.show()
def plot_feas(data):
    fig, axes = plt.subplots(nrows=2, ncols=4)
    axes = np.ndarray.flatten(np.array(axes))
    fig2, axes2 = plt.subplots(nrows=2, ncols=4)
    axes2 = np.ndarray.flatten(np.array(axes2))
    for (i, (theta, feasible)) in enumerate(data):
        ax = axes[i]
        ax2 = axes2[i]
        # data_means = np.array(dist.particles)[:,:2].T
        # kernel = kde(data_means)
        # xx, yy = np.mgrid[-1.75:0.5:100j, -1.25:1.25:100j]
        # positions = np.vstack([xx.ravel(), yy.ravel()])
        # f = np.reshape(kernel(positions).T, xx.shape)
        # cfset = ax.contourf(xx, yy, f, cmap='Greens')
        # cset = ax.contour(xx, yy, f, colors='k')
        # ax.clabel(cset, inline=1, fontsize=10)
        ax.scatter(feasible[:,0], feasible[:,1], c='C0')
        ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
        ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)

        # data_means = np.array(dist.particles)[:,2:4].T
        # kernel = kde(data_means)
        # xx, yy = np.mgrid[-2:1:100j, -2.5:0.5:100j]
        # positions = np.vstack([xx.ravel(), yy.ravel()])
        # f = np.reshape(kernel(positions).T, xx.shape)
        # cfset = ax2.contourf(xx, yy, f, cmap='Greens')
        # cset = ax2.contour(xx, yy, f, colors='k')
        # ax2.clabel(cset, inline=1, fontsize=10)
        ax2.scatter(feasible[:,2], feasible[:,3], c='C0')
        ax2.scatter(theta[2], theta[3], c='C2', s=200, zorder=2)
        ax2.scatter(TRUE_MEAN[2], TRUE_MEAN[3], c='C3', s=200, zorder=2)
        plt.pause(0.2)
    fig.suptitle("Feasible sets to choose from (dim 1&2)")
    fig2.suptitle("Feasible sets to choose from (dim 3&4)")
#########################################################
all_data = np.load('./4joint_sim_training_data_mean0.npy')
# shuffle(all_data)
data = all_data
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
dist = SetWeightsParticleDistribution(particles, weights, cost, w=TRUE_WEIGHTS, ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)
# (theta, feasible) = data[0]
# dist.weights = dist.reweight(theta, feasible)
# dist.resample()
# dist.weights = dist.reweight(theta, feasible)
# dist.resample()

# plot_feas(data)
# plt.show()
def info_gain(dist, x):
    return (x, dist.info_gain(x[1], num_boxes=20))
# if __name__ == '__main__':
#     pool = mp.Pool(8)
fig, axes = plt.subplots(nrows=5, ncols=5)
axes = np.ndarray.flatten(np.array(axes))
fig2, axes2 = plt.subplots(nrows=5, ncols=5)
axes2 = np.ndarray.flatten(np.array(axes2))
# bar_fig, bar_axes = plt.subplots(nrows=4, ncols=3)
# bar_axes = np.ndarray.flatten(np.array(bar_axes))
fig.suptitle('dim 1&2 Particles: ' + str(NUM_PARTICLES) + ' alpha_i: ' + str(ALPHA_I) +\
             ' alpha_o: ' + str(ALPHA_O))
fig2.suptitle('dim 3&4 Particles: ' + str(NUM_PARTICLES) + ' alpha_i: ' + str(ALPHA_I) +\
             ' alpha_o: ' + str(ALPHA_O))
ax = axes[0]
ax2 = axes2[0]
data_means = np.array(dist.particles)[:,:2].T
kernel = kde(data_means)
xx, yy = np.mgrid[-1.75:0.5:100j, -1.25:1.25:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
f = np.reshape(kernel(positions).T, xx.shape)
cfset = ax.contourf(xx, yy, f, cmap='Greens')
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)

data_means = np.array(dist.particles)[:,2:4].T
kernel = kde(data_means)
xx, yy = np.mgrid[-2:1:100j, -2.5:0.5:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
f = np.reshape(kernel(positions).T, xx.shape)
cfset = ax2.contourf(xx, yy, f, cmap='Greens')
cset = ax2.contour(xx, yy, f, colors='k')
ax2.clabel(cset, inline=1, fontsize=10)
ax2.scatter(TRUE_MEAN[2], TRUE_MEAN[3], c='C3', s=200, zorder=2)
plt.pause(0.2)
for i in range(1, 25):
    # func = partial(info_gain, dist)
    # pooled = pool.map(func, data)
    # # pooled = [func(x) for x in data]
    # print
    # expected_infos = [sample[1] for sample in pooled]
    # max_idx = np.argmax(expected_infos)
    # (theta, feasible) = pooled[max_idx][0]
    # actual_infos = []
    # ent_before = dist.entropy(num_boxes=20)
    # for j in range(len(data)):
    #     t, f = data[j]
    #     d = SetWeightsParticleDistribution(dist.particles, dist.weights, dist.cost, dist.w, dist.ALPHA_I, dist.ALPHA_O)
    #     d.weights = d.reweight(t, f)
    #     actual_infos.append(ent_before - d.entropy(num_boxes=20))
    (theta, feasible) = data[i]
    dist.weights = dist.reweight_vectorized(theta, feasible)
    dist.resample()

    ax = axes[i]
    ax2 = axes2[i]
    # bar_ax = bar_axes[i]
    data_means = np.array(dist.particles)[:,:2].T
    kernel = kde(data_means)
    xx, yy = np.mgrid[-1.75:0.5:100j, -1.25:1.25:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='Greens')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
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
    ax2.scatter(feasible[:,2], feasible[:,3], c='C0')
    ax2.scatter(theta[2], theta[3], c='C2', s=200, zorder=2)
    ax2.scatter(TRUE_MEAN[2], TRUE_MEAN[3], c='C3', s=200, zorder=2)

    # bar_ax.bar(np.arange(len(data)), expected_infos, 0.35, color='C0', label='expected info gain')
    # bar_ax.bar(np.arange(len(data)) + 0.35, actual_infos, 0.35, color='C1', label='actual info gain')
    # bar_ax.bar(max_idx, expected_infos[max_idx], 0.35, color='C2', label='chosen set expected info')
    plt.pause(0.2)
plt.show()
