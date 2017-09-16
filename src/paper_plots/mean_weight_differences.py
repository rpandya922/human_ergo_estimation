from __future__ import division
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../weight_learning')
import utils
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde as kde
import probability_estimation as pe
from distribution import SetWeightsParticleDistribution
import multiprocessing as mp
from functools import partial
from sklearn.neighbors import NearestNeighbors
from numpy.random import multivariate_normal as mvn
from tqdm import tqdm
###################################################################
# CONSTANTS/FUNCTIONS
np.random.seed(0)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
seaborn.set_style('ticks')
DOF = 2
l1 = 3
l2 = 3
ALPHA = 10
ALPHA_I = 1
ALPHA_O = 1
TRUE_MEAN = np.array([0, 0])
TRUE_WEIGHTS = np.array([1, 1])
TRUE_WEIGHTS = TRUE_WEIGHTS / np.linalg.norm(TRUE_WEIGHTS)
NUM_PARTICLES = 500
two_pi = 2 * np.pi
FEASIBLE_COLOR = '#67a9cf'
GROUND_TRUTH_COLOR = '#1b9e77'
PICKED_COLOR = '#ceb301'
FIG_SIZE = (15, 12)
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
def get_theta_old(x, y):
    theta2 = np.arccos(((l1**2) + (l2**2) - (x**2) - (y**2)) / (2 * l1 * l2))
    theta1 = np.arcsin(l2 * np.sin(theta2) / np.sqrt((x**2) + (y**2))) +\
             np.arctan(y / x)
    if np.isnan(theta2) or np.isnan(theta1):
        return []
    return [[theta1, theta2], [-theta1, theta2]]
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
def create_sample(feas):
    nums = pe.prob_theta_given_lam_stable_set_weight_num(feas, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA)
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feas, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA)
    probs = np.exp(nums - denom)
    chosen_idx = np.argmax(probs)
    chosen = feas[chosen_idx]
    return (chosen, feas)
def create_sample_from_xy(obj):
    feas = []
    for (x, y) in obj:
        feas.extend(get_theta(x, y))
    feas = np.array(feas)
    nums = pe.prob_theta_given_lam_stable_set_weight_num(feas, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA)
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feas, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA)
    probs = np.exp(nums - denom)
    chosen_idx = np.argmax(probs)
    chosen = feas[chosen_idx]
    return (chosen, feas)
def plot_objects(data):
    fig, axes = plt.subplots(nrows=2, ncols=4)
    fig.suptitle("'objects' in xy space")
    axes = np.ndarray.flatten(np.array(axes))
    for (i, feasible) in enumerate(data):
        ax = axes[i]
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.scatter(feasible[:,0], feasible[:,1])
    plt.pause(0.2)
def plot_feas(data):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=FIG_SIZE, subplot_kw={'aspect': 'equal'})
    fig.suptitle("feasible sets in theta space, l1: 3, l2: 3")
    axes = np.ndarray.flatten(np.array(axes))
    for (i, (theta, feasible)) in enumerate(data):
        ax = axes[i]
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.scatter(feasible[:,0], feasible[:,1], c=FEASIBLE_COLOR)
    plt.pause(0.2)
def plot_belief(ax, particles, ground_truth, i=1):
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    data_means = particles.T
    kernel = kde(data_means)
    xx, yy = np.mgrid[-6:6:100j, -6:6:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='gist_gray_r')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.scatter(ground_truth[0], ground_truth[1], c='C3', s=200, zorder=2, label=r"$\theta^{*}$")
    if i != 0:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')
def plot_belief_update(ax, particles, theta, feasible, ground_truth, i=1):
    plot_belief(ax, particles, ground_truth, i)
    ax.scatter(feasible[:,0], feasible[:,1], c='C0', label=r'$\Theta_{feas}$')
    ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2, label=r'$\theta_H$')
    if i == 1:
        lgnd = ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left")
        lgnd.legendHandles[0]._sizes = [30]
        lgnd.legendHandles[1]._sizes = [30]
        lgnd.legendHandles[2]._sizes = [30]
def plot_likelihood_heatmap(ax, theta, feasible, ground_truth, with_belief=False, dist=None, vmin=None, vmax=None):
    xx, yy = np.mgrid[-5:5:100j, -5:5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feasible)
    distances, indices = nbrs.kneighbors(feasible)
    max_dist = min(np.amax(distances), 0.5)
    distances, indices = nbrs.kneighbors(positions.T)
    def likelihood(idx, point):
        if distances[idx][0] >= max_dist:
            alpha = ALPHA_O
        else:
            alpha = ALPHA_I
        return np.exp(pe.prob_theta_given_lam_stable_set_weight_num(theta, point, TRUE_WEIGHTS, cost, alpha)\
        -pe.prob_theta_given_lam_stable_set_weight_denom(feasible, point, TRUE_WEIGHTS, cost, alpha))
    likelihoods = np.array([likelihood(idx, p) for idx, p in enumerate(positions.T)])
    print np.amin(likelihoods)
    print "max: " + str(np.amax(likelihoods))
    f = np.reshape(likelihoods.T, xx.shape)
    ax.imshow(np.flip(f, 1).T, cmap='gist_gray_r', interpolation='nearest', extent=(-5, 5, -5, 5), vmin=vmin, vmax=vmax)
    ax.scatter(feasible[:,0], feasible[:,1], c=FEASIBLE_COLOR)
    ax.scatter(theta[0], theta[1], c=PICKED_COLOR, s=200, zorder=2)
    if with_belief:
        data_means = np.array(dist.particles).T
        kernel = kde(data_means)
        xx, yy = np.mgrid[-5:5:100j, -5:5:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)
        cset = ax.contour(xx, yy, f, colors='k')
    ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c=GROUND_TRUTH_COLOR, s=200, zorder=2)
def plot_likelihood_heatmap_expectation(ax, feasible, ground_truth, with_belief=False, dist=None, vmin=None, vmax=None):
    xx, yy = np.mgrid[-5:5:50j, -5:5:50j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feasible)
    distances, indices = nbrs.kneighbors(feasible)
    max_dist = min(np.amax(distances), 0.5)
    distances, indices = nbrs.kneighbors(positions.T)
    def likelihood(idx, point):
        if distances[idx][0] >= max_dist:
            alpha = ALPHA_O
        else:
            alpha = ALPHA_I
        l = 0
        for theta in feasible:
            l += np.exp(pe.prob_theta_given_lam_stable_set_weight_num(theta, point, TRUE_WEIGHTS, cost, alpha)\
                 -pe.prob_theta_given_lam_stable_set_weight_denom(feasible, point, TRUE_WEIGHTS, cost, alpha))
        return (l / len(feasible))
    likelihoods = np.array([likelihood(idx, p) for idx, p in enumerate(positions.T)])
    print np.amin(likelihoods)
    print "max: " + str(np.amax(likelihoods))
    f = np.reshape(likelihoods.T, xx.shape)
    ax.imshow(np.flip(f, 1).T, cmap='gist_gray_r', interpolation='nearest', extent=(-5, 5, -5, 5), vmin=vmin, vmax=vmax)
    ax.scatter(feasible[:,0], feasible[:,1], c=FEASIBLE_COLOR)
    if with_belief:
        data_means = np.array(dist.particles).T
        kernel = kde(data_means)
        xx, yy = np.mgrid[-5:5:100j, -5:5:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)
        cset = ax.contour(xx, yy, f, colors='k')
    ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c=GROUND_TRUTH_COLOR, s=200, zorder=2)
def sample_spherical(npoints, ndim=2):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec
###################################################################
# data_original = [create_box([-1, 3], [-1, -3], 0.1)]
# data_original = [sample_spherical(700).T]
data_original = [create_box([-1, 3], [-1, -3], 0.1), utils.create_line([-3, 3], [3, -3], 0.01),\
                 sample_spherical(200).T, sample_spherical(200).T, sample_spherical(200).T, \
                 sample_spherical(200).T, sample_spherical(200).T, create_box([-2, 2], [2, -2])]
# data_original = [create_box([-3, 2.5], [0, 1.5]), create_box([-3, 1], [0, 0]),\
#                  create_box([-3, -0.5], [0, -1.5]), create_box([-3, -2], [0, -3]),\
#                  create_box([0.5, 2.5], [3.5, 1.5]), create_box([0.5, 1], [3.5, 0]),\
#                  create_box([0.5, -0.5], [3.5, -1.5]), create_box([0.5, -2], [3.5, -3])]
# data_original = [create_box([-1, 3], [-1, -3], 0.1), create_box([-1, 3], [-1, -3]),\
#                  create_ellipse(0, 2.5, 1, 2), create_box([-3, 2.5], [0, 1.5]),\
#                  create_ellipse(0, 0, 2, 2), create_box([-3, 2.5], [0, 1.5], 0.1),\
#                  create_ellipse(0, 1, 1, 2), create_ellipse(-1, 0, 1, 2)]
# cov = np.array([[0.1, 0], [0, 0.1]])
# data_original = [ create_box([-2, -2], [3, -2]), create_box([-1, 3], [-1, -3], 0.1),\
#                  create_box([-1, 3], [-1, -3], 0.5), np.array([[3, 3], [-3, 2]]),\
#                  np.vstack((mvn([3, 3], cov, 10), mvn([-3, 2], cov, 10))),\
#                  np.vstack((mvn([3, 3], cov, 10), mvn([-3, 2], cov, 10), mvn([0, -3], cov, 10))),\
#                  create_box([0, 3], [2.5, 1], 0.1), create_box([0, 3], [2.5, 1], 0.5),\
#                  create_box([0, 3], [4.5, 2])]
data = [create_sample(feas) for feas in data_original]
# data_original = [create_ellipse(1, 1, 1, 1), create_ellipse(0, 0, 1, 2), \
#         create_box([1, 3], [5, 0]), create_box([0, 1], [4, 1]), \
#         create_ellipse(0, -2, 1, 3), \
#         create_box([-6, 0], [6, 0]), create_ellipse(-3, -3, 2, 2)]
# data = [create_sample_from_xy(shape) for shape in data_original]
# plot_objects(data_original)
plot_feas(data)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=FIG_SIZE, subplot_kw={'aspect': 'equal'})
axes = np.ndarray.flatten(np.array(axes))
for (i, (theta, feasible)) in enumerate(data):
    ax = axes[i]
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # plot_likelihood_heatmap(ax, theta, feasible, TRUE_MEAN, vmin=-22, vmax=-2.5)
    plot_likelihood_heatmap(ax, theta, feasible, TRUE_MEAN, vmin=0)
    plt.pause(0.1)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=FIG_SIZE, subplot_kw={'aspect': 'equal'})
axes = np.ndarray.flatten(np.array(axes))
for (i, (theta, feasible)) in enumerate(data):
    ax = axes[i]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    utils.plot_likelihood_heatmap_norm_weights(ax, theta, feasible, TRUE_WEIGHTS, \
    vmin=0, color=GROUND_TRUTH_COLOR)
    plt.pause(0.1)
plt.show()
