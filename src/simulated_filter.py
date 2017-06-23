from __future__ import division
import seaborn
import numpy as np
from scipy.optimize import minimize
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import probability_estimation as pe
from distribution import ParticleDistribution

#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 4
NUM_PARTICLES = 50
h = 0.2
weight_widths = 10
theta_widths = 3.14
num_boxes = 5
axis_ranges = 2 * np.array([weight_widths, weight_widths, weight_widths, weight_widths, theta_widths, theta_widths, theta_widths, theta_widths])
#########################################################
data = np.load('./sim_training_data.npy')[:15]

def cost(theta, theta_star, w):
    return pe.distance_cost(theta, theta_star, w)
def prior(vec):
    return 1

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

expected_info = []
expected_info2 = []
actual_info = []
actual_info2 = []
(theta1, Theta_x1) = data[0]
for (theta, Theta_x) in data:
    print len(theta_x)
    expected = dist.info_gain(Theta_x, num_boxes, axis_ranges)
    expected_info.append(expected)
    # expected2 = dist.info_gain(Theta_x1, num_boxes, axis_ranges)
    # expected_info2.append(expected2)
    print "Expected info gain: " + str(expected)
    entropy = dist.entropy(num_boxes, axis_ranges)
    # print entropy
    dist.weights = dist.reweight(theta, Theta_x)
    # dist2.weights = dist.reweight(theta, Theta_x)
    # dist2.resample()
    print "reweighted"
    dist.resample()
    actual = dist.entropy(num_boxes, axis_ranges) - entropy
    # actual_info2.append(dist2.entropy(num_boxes, axis_ranges) - entropy)
    # dist2 = ParticleDistribution(dist.particles, dist.weights, dist.cost)
    print "Actual info gain: " + str(actual)
    actual_info.append(actual)

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
    # entropy = dist.entropy(num_boxes, axis_ranges)
    # print "Entropy: " + str(entropy)
    print

mean = np.mean(actual_info)
tss = np.sum(np.square(np.array(actual_info) - mean))
rss = np.sum(np.square(np.array(actual_info) - np.array(expected_info)))
r_squared = 1 - rss / tss
print "R^2: "  + str(r_squared)

# mean = np.mean(actual_info2)
# tss = np.sum(np.square(np.array(actual_info2) - mean))
# rss = np.sum(np.square(np.array(actual_info2) - np.array(expected_info)))
# r_squared = 1 - rss / tss
# print "R^2 new: "  + str(r_squared)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('iteration')
ax.set_ylabel('information gain')
ax.plot(range(len(data)), expected_info, label='expected')
# ax.plot(range(len(data)), expected_info2, label='expected same sample')
ax.plot(range(len(data)), actual_info, label='actual')
# ax.plot(range(len(data)), actual_info2, label='actual new sample')
plt.legend(loc='upper left')
plt.show()