from __future__ import division
import seaborn
import numpy as np
from scipy.optimize import minimize
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import probability_estimation as pe
from distribution import ParticleDistribution
from random import shuffle
import multiprocessing as mp

#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 4
NUM_PARTICLES = 20
h = 0.2
weight_widths = 10
theta_widths = 3.14
num_boxes = 5
axis_ranges = 2 * np.array([weight_widths, weight_widths, weight_widths, weight_widths, theta_widths, theta_widths, theta_widths, theta_widths])
def cost(theta, theta_star, w):
    return pe.distance_cost(theta, theta_star, w)
def prior(vec):
    return 1
data = np.load('./sim_training_data.npy')
shuffle(data)
data = data[:12]
particles = []
weights = []
for i in range(NUM_PARTICLES):
    lam = np.random.uniform(-weight_widths, weight_widths, (DOF,))
    lam = np.hstack((lam, np.random.uniform(-theta_widths, theta_widths, (DOF,))))
    weight = 1
    theta = lam[DOF:]
    particles.append(lam)
    weights.append(weight)
weights = np.array(weights) / np.sum(weights)

dist = ParticleDistribution(particles, weights, cost)
dist2 = ParticleDistribution(particles, weights, cost)
(theta1, feas1) = data[0]
def info_gain(x):
    return (x, dist2.info_gain(x[1], num_boxes, axis_ranges))
#########################################################
if __name__ == '__main__':
    pool = mp.Pool(5)
    same_entropy = []
    max_info_entropy = []
    same_entropy.append(dist.entropy(num_boxes, axis_ranges))
    max_info_entropy.append(dist2.entropy(num_boxes, axis_ranges))
    for i in range(5):
        dist.weights = dist.reweight(theta1, feas1)
        dist.resample()
        same_entropy.append(dist.entropy(num_boxes, axis_ranges))
        (theta_max, feas_max) = min(pool.map(info_gain, data), key=lambda x: x[1])[0]
        dist2.weights = dist2.reweight(theta_max, feas_max)
        dist2.resample()
        max_info_entropy.append(dist2.entropy(num_boxes, axis_ranges))
        print i

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('iteration')
    ax.set_ylabel('entropy')
    ax.plot(range(len(same_entropy)), same_entropy, label='same sample')
    ax.plot(range(len(max_info_entropy)), max_info_entropy, label='max info sample')
    plt.legend(loc='upper left')
    plt.show()