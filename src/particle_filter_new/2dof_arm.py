from __future__ import division
import sys
sys.path.insert(0, '../')
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
import probability_estimation as pe
from distribution import SetWeightsParticleDistribution
import multiprocessing as mp
from functools import partial
###################################################################
# CONSTANTS/FUNCTIONS
DOF = 2
l1 = 3
l2 = 3
box_size = 0.5
ALPHA = 10
ALPHA_I = 2
ALPHA_O = 1
TRUE_MEAN = np.array([0, 0])
TRUE_WEIGHTS = np.array([1, 1])
NUM_PARTICLES = 500
two_pi = 2 * np.pi
def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)
def create_box(upper_left, lower_right):
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
    rand = np.random.uniform(0, 1, size=(1000, 2)) * np.array([x0 + 2*a, y0 + 2*b])
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
    if thetas[0][0] > np.pi:
        thetas[0][0] -= two_pi
    elif thetas[0][0] < -np.pi:
        thetas[0][0] += two_pi
    if thetas[1][0] > np.pi:
        thetas[1][0] -= two_pi
    elif thetas[1][0] < -np.pi:
        thetas[1][0] += two_pi
    return thetas
def create_sample(obj):
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
###################################################################
data_original = [create_ellipse(1, 1, 1, 1), create_ellipse(0, 0, 1, 2), \
        create_box([1, 3], [5, 0]), create_box([0, 1], [4, 1]), \
        create_ellipse(0, -2, 1, 3), \
        create_box([-6, 0], [6, 0]), create_ellipse(-3, -3, 2, 2)]
data = [create_sample(shape) for shape in data_original]
fig, axes = plt.subplots(nrows=2, ncols=4)
fig.suptitle("feasible sets in theta space, l1: 3, l2: 3")
axes = np.ndarray.flatten(np.array(axes))
for (i, (theta, feasible)) in enumerate(data):
    ax = axes[i]
    ax.set_xlim(-3.14, 3.14)
    ax.set_ylim(-3.14, 3.14)
    ax.scatter(feasible[:,0], feasible[:,1])
plt.pause(0.2)
fig, axes = plt.subplots(nrows=2, ncols=4)
fig.suptitle("'objects' in xy space")
axes = np.ndarray.flatten(np.array(axes))
for (i, feasible) in enumerate(data_original):
    ax = axes[i]
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.scatter(feasible[:,0], feasible[:,1])
plt.pause(0.2)

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

def info_gain(dist, x):
    return (x, dist.info_gain(x[1], num_boxes=20), dist.expected_cost(x[1]))
if __name__ == '__main__':
    pool = mp.Pool(8)
    fig, axes = plt.subplots(nrows=4, ncols=3)
    axes = np.ndarray.flatten(np.array(axes))
    bar_fig, bar_axes = plt.subplots(nrows=4, ncols=3)
    bar_axes = np.ndarray.flatten(np.array(bar_axes))
    fig.suptitle('Particles: ' + str(NUM_PARTICLES) + ' alpha_i: ' + str(ALPHA_I) +\
                 ' alpha_o: ' + str(ALPHA_O) + ' l1: ' + str(l1) + ' l2: ' +str(l2) +\
                 ' cost weights: ' + str(TRUE_WEIGHTS))
    ax = axes[0]
    data_means = np.array(dist.particles)[:,:2].T
    kernel = kde(data_means)
    xx, yy = np.mgrid[-5:5:100j, -5:5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='Greens')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)
    plt.pause(0.1)

    for i in range(1, len(axes)):
        func = partial(info_gain, dist)
        pooled = pool.map(func, data)
        print
        expected_infos = [sample[1] for sample in pooled]
        expected_costs = [sample[2] for sample in pooled]
        max_idx = np.argmax(expected_infos)
        # max_idx = np.argmin(expected_costs)
        (theta, feasible) = pooled[max_idx][0]
        actual_infos = []
        ent_before = dist.entropy(num_boxes=20)
        for j in range(len(data)):
            t, f = data[j]
            d = SetWeightsParticleDistribution(dist.particles, dist.weights, dist.cost, dist.w, dist.ALPHA_I, dist.ALPHA_O)
            d.weights = d.reweight(t, f)
            actual_infos.append(ent_before - d.entropy(num_boxes=20))
        dist.weights = dist.reweight(theta, feasible)
        dist.resample()

        ax = axes[i]
        data_means = np.array(dist.particles)[:,:2].T
        kernel = kde(data_means)
        xx, yy = np.mgrid[-5:5:100j, -5:5:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)
        cfset = ax.contourf(xx, yy, f, cmap='Greens')
        cset = ax.contour(xx, yy, f, colors='k')
        ax.scatter(feasible[:,0], feasible[:,1], label='feasible set')
        ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
        ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)

        bar_ax = bar_axes[i]
        bar_ax.bar(np.arange(len(data)), expected_infos, 0.35, color='C0', label='expected info gain')
        bar_ax.bar(np.arange(len(data)) + 0.35, actual_infos, 0.35, color='C1', label='actual info gain')
        bar_ax.bar(max_idx, expected_infos[max_idx], 0.35, color='C2', label='chosen set expected info')
        plt.pause(0.1)
    plt.show()