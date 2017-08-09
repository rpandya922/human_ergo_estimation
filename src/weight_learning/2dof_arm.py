from __future__ import division
import sys
sys.path.insert(0, '../')
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
import probability_estimation as pe
from distribution import SetMeanParticleDistribution
import multiprocessing as mp
from functools import partial
from sklearn.neighbors import NearestNeighbors
from numpy.random import multivariate_normal as mvn
###################################################################
# CONSTANTS/FUNCTIONS
DOF = 2
l1 = 3
l2 = 3
ALPHA = 10
ALPHA_I = 1
ALPHA_O = 1
TRUE_MEAN = np.array([0, 0])
TRUE_WEIGHTS = np.array([1, 0])
TRUE_WEIGHTS = TRUE_WEIGHTS / np.linalg.norm(TRUE_WEIGHTS)
NUM_PARTICLES = 500
two_pi = 2 * np.pi
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
    # thetas = [[theta1_partial + theta_prime, -theta2], [theta1_partial - theta_prime, theta2]]
    thetas = [[theta1_partial - theta_prime, theta2]]
    if thetas[0][0] > np.pi:
        thetas[0][0] -= two_pi
    elif thetas[0][0] < -np.pi:
        thetas[0][0] += two_pi
    # if thetas[1][0] > np.pi:
    #     thetas[1][0] -= two_pi
    # elif thetas[1][0] < -np.pi:
    #     thetas[1][0] += two_pi
    return thetas
def create_sample(feas):
    nums = pe.prob_theta_given_lam_stable_set_weight_num(feas, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA)
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feas, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA)
    probs = np.exp(nums - denom)
    chosen_idx = np.argmax(probs)
    chosen = feas[chosen_idx]
    return (chosen, feas)
def create_sample2(feas, weights):
    nums = pe.prob_theta_given_lam_stable_set_weight_num(feas, TRUE_MEAN, weights, cost, ALPHA)
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feas, TRUE_MEAN, weights, cost, ALPHA)
    probs = np.exp(nums - denom)
    chosen_idx = np.argmax(probs)
    chosen = feas[chosen_idx]
    return (chosen, feas)
def plot_feas(data, weights=None):
    if weights is None:
        weights = [TRUE_WEIGHTS] * len(data)
    fig, axes = plt.subplots(nrows=2, ncols=4)
    fig.suptitle("feasible sets in theta space, l1: 3, l2: 3")
    axes = np.ndarray.flatten(np.array(axes))
    for (i, (theta, feasible)) in enumerate(data):
        w = weights[i]
        ax = axes[i]
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.scatter(feasible[:,0], feasible[:,1])
        ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
        ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)
        ax.set_title('weights: ' + str(w))
    plt.pause(0.2)
def plot_belief(ax, particles, ground_truth):
    data_means = particles.T
    kernel = kde(data_means)
    xx, yy = np.mgrid[0:1.25:100j, 0:1.25:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='Greens')
    cset = ax.contour(xx, yy, f, colors='k')
    # ax.clabel(cset, inline=1, fontsize=10)
    ax.scatter(ground_truth[0], ground_truth[1], c='C3', s=200, zorder=2)
def plot_belief_update(ax, particles, theta, feasible, ground_truth):
    plot_belief(ax, particles, ground_truth)
    ax.scatter(feasible[:,0], feasible[:,1], c='C0')
    ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
def get_min_max_likelihood(data):
    mins = []
    maxes = []
    for (theta, feasible) in data:
        xx, yy = np.mgrid[0:1.25:100j, 0:1.25:100j]
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
            return pe.prob_theta_given_lam_stable_set_weight_num(theta, TRUE_MEAN, point, cost, alpha)\
            -pe.prob_theta_given_lam_stable_set_weight_denom(feasible, TRUE_MEAN, point, cost, alpha)
        likelihoods = np.array([likelihood(idx, p) for idx, p in enumerate(positions.T)])
        mins.append(np.amin(likelihoods))
        maxes.append(np.amax(likelihoods))
    return min(mins), max(maxes)
def plot_likelihood_heatmap(ax, theta, feasible, ground_truth, vmin=-27, vmax=-2, with_belief=False, dist=None):
    xx, yy = np.mgrid[0:1.25:100j, 0:1.25:100j]
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
        return pe.prob_theta_given_lam_stable_set_weight_num(theta, TRUE_MEAN, point, cost, alpha)\
        -pe.prob_theta_given_lam_stable_set_weight_denom(feasible, TRUE_MEAN, point, cost, alpha)
    likelihoods = np.array([likelihood(idx, p) for idx, p in enumerate(positions.T)])
    f = np.reshape(likelihoods.T, xx.shape)
    ax.imshow(np.flip(f, 1).T, cmap='inferno', interpolation='nearest', extent=(0, 1.25, 0, 1.25), vmin=vmin, vmax=vmax)
    if with_belief:
        data_means = np.array(dist.particles).T
        kernel = kde(data_means)
        xx, yy = np.mgrid[0:1.25:100j, 0:1.25:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)
        cset = ax.contour(xx, yy, f, colors='k')
    ax.scatter(ground_truth[0], ground_truth[1], c='C3', s=200, zorder=2)
###################################################################
data_original = [create_ellipse(-1, 2, 1, 2), create_ellipse(-1, -2, 1, 2),\
                 create_ellipse(1, 2, 1, 2), create_ellipse(1, -2, 1, 2), \
                 create_ellipse(2, 2, 2, 1), create_ellipse(-2, -2, 2, 1), \
                 create_ellipse(-1, 0, 1, 3), create_ellipse(0, 0, 1, 1)]
# data_original = [create_ellipse(-2, -2, 2, 1.5)]
data = [create_sample(feas) for feas in data_original]
plot_feas(data)

particles = []
weights = []
all_feas = []
for (theta, feasible) in data:
    all_feas.extend(feasible)
all_feas = np.array(all_feas)
mins = np.array([0, 0])
maxes = np.array([1, 1])
ranges = maxes - mins
particles = np.random.uniform(0, 1, size=(NUM_PARTICLES, DOF))
particles *= ranges
particles += mins
weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES
weights = np.array(weights) / np.sum(weights)
dist = SetMeanParticleDistribution(particles, weights, cost, m=TRUE_MEAN, \
ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=0)
dist.resample()
fig, axes = plt.subplots(nrows=2, ncols=4)
axes = np.ndarray.flatten(np.array(axes))
vmin, vmax = get_min_max_likelihood(data)
for i, (theta, feasible) in enumerate(data):
    plot_likelihood_heatmap(axes[i], theta, feasible, TRUE_WEIGHTS, vmin, vmax)
    plt.pause(0.1)

fig, axes = plt.subplots(nrows=5, ncols=5)
axes = np.ndarray.flatten(np.array(axes))
plot_belief(axes[0], np.array(dist.particles), TRUE_WEIGHTS)
plt.pause(0.1)
for i in range(50):
    ax = axes[(i+1) % len(axes)]
    ax.cla()
    theta, feasible = data[i % len(data)]
    dist.weights = dist.reweight(theta, feasible)
    dist.resample()
    plot_belief(ax, np.array(dist.particles), TRUE_WEIGHTS)
    plt.pause(0.1)
plt.show()
