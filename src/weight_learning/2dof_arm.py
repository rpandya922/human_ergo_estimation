from __future__ import division
import sys
sys.path.insert(0, '../')
import numpy as np
import seaborn
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
import probability_estimation as pe
from distribution import SetMeanParticleDistribution
import multiprocessing as mp
from functools import partial
from sklearn.neighbors import NearestNeighbors
from numpy.random import multivariate_normal as mvn
import plotting_utils as pu
###################################################################
# CONSTANTS/FUNCTIONS
DOF = 2
l1 = 3
l2 = 3
ALPHA = 10
ALPHA_I = 1
ALPHA_O = 1
TRUE_MEAN = np.array([0, 0])
TRUE_WEIGHTS = np.array([1, 3])
TRUE_WEIGHTS = TRUE_WEIGHTS / np.linalg.norm(TRUE_WEIGHTS)
NUM_PARTICLES = 500
two_pi = 2 * np.pi
def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)
###################################################################
# data_original = [create_ellipse(-1, 2, 1, 2), create_ellipse(-1, -2, 1, 2),\
#                  create_ellipse(1, 2, 1, 2), create_ellipse(1, -2, 1, 2), \
#                  create_ellipse(2, 2, 2, 1), create_ellipse(-2, -2, 2, 1), \
#                  create_ellipse(-1, 0, 1, 3), create_ellipse(0, 0, 1, 1)]
# cov = np.array([[0.1, 0], [0, 0.1]])
# data_original = [np.array([[1, 0], [0, 1]]), np.vstack((mvn([2, 1], cov, 10), mvn([-3, 2], cov, 10))),\
#                  create_line([1, 0], [0, 1]), create_line([-1, 2], [2, -1]),\
#                  create_line([1, 3], [-3, -1]), np.array([[1, 3], [-3, -1]]),\
#                  create_line([-2, -2], [2, -2])]
data_original = [np.array([[1, 0], [0, 1]]), pu.create_line([1, 0], [0, 1]),\
                 pu.create_line([-1, 2], [2, -1]), pu.create_line([-2, 0], [0, 3]),\
                 pu.create_line([-1, -1], [1, -1]), pu.create_box([1, 1], [1, -1], 0.1),\
                 pu.create_line([-3, 1], [1, -3]), pu.create_line([-1, -5], [5, 1])]
data = [pu.create_sample(feas) for feas in data_original]
pu.plot_feas(data)
# data_original = [pu.create_ellipse(1, 1, 1, 1), pu.create_ellipse(0, 0, 1, 2), \
#         pu.create_box([1, 3], [5, 0]), pu.create_box([0, 1], [4, 1]), \
#         pu.create_ellipse(0, -2, 1, 3), \
#         pu.create_box([-6, 0], [6, 0]), pu.create_ellipse(-3, -3, 2, 2)]
# data = [pu.create_sample_from_xy(shape) for shape in data_original]
# pu.plot_objects(data_original)
# pu.plot_feas(data)

particles = []
weights = []
while len(particles) < NUM_PARTICLES:
    p = np.random.randn(DOF, 1).T[0]
    p = p / np.linalg.norm(p, axis=0)
    if p[0] >= 0 and p[1] >= 0:
        particles.append(p)
particles = np.array(particles)
weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES
dist = SetMeanParticleDistribution(particles, weights, cost, m=TRUE_MEAN, \
ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=0.01)
dist.resample()
fig, axes = plt.subplots(nrows=3, ncols=3)
axes = np.ndarray.flatten(np.array(axes))
# vmin, vmax = get_min_max_likelihood(data)
for i, (theta, feasible) in enumerate(data):
    pu.plot_likelihood_heatmap_norm_weights(axes[i], theta, feasible, TRUE_WEIGHTS)
    plt.pause(0.1)

def info_gain(dist, x):
    return (x, dist.info_gain(x[1], num_boxes=20), dist.expected_cost(x[1]))
if __name__ == '__main__':
    pool = mp.Pool(8)
    fig, axes = plt.subplots(nrows=5, ncols=4)
    axes = np.ndarray.flatten(np.array(axes))
    bar_fig, bar_axes = plt.subplots(nrows=5, ncols=4)
    bar_axes = np.ndarray.flatten(np.array(bar_axes))
    pu.plot_belief(axes[0], np.array(dist.particles), TRUE_WEIGHTS)
    plt.pause(0.1)
    for i in range(1, len(axes)):
        ax = axes[(i) % len(axes)]
        bar_ax = bar_axes[i]
        ax.cla()
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
            d = SetMeanParticleDistribution(dist.particles, dist.weights, dist.cost, dist.m, dist.ALPHA_I, dist.ALPHA_O)
            d.weights = d.reweight(t, f)
            actual_infos.append(ent_before - d.entropy(num_boxes=20))
        # theta, feasible = data[i % len(data)]
        dist.weights = dist.reweight(theta, feasible)
        dist.resample()
        pu.plot_belief(ax, np.array(dist.particles), TRUE_WEIGHTS)
        bar_ax.bar(np.arange(len(data)), expected_infos, 0.35, color='C0', label='expected info gain')
        bar_ax.bar(np.arange(len(data)) + 0.35, actual_infos, 0.35, color='C1', label='actual info gain')
        bar_ax.bar(max_idx, expected_infos[max_idx], 0.35, color='C2', label='chosen set expected info')
        plt.pause(0.1)
    plt.show()
