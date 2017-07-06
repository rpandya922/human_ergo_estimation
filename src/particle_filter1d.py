from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import probability_estimation as pe
from distribution import SetWeightsParticleDistribution

#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 1
NUM_PARTICLES = 1000
h = 0.2
weight_widths = 10
theta_widths = 5
num_boxes = 100
axis_ranges = 2 * np.array([weight_widths])
box_size = theta_widths * 2 / num_boxes
def cost(theta, theta_star, w):
    return pe.distance_cost(theta, theta_star, w)
def prior(vec):
    return 1
def create_feasible_set(left, right):
    feas = []
    x = -theta_widths
    while x <= right:
        if x >= left:
            feas.append([x])
        x += box_size
    return np.array(feas)
#########################################################
particles = []
weights = []
for i in range(NUM_PARTICLES):
    lam = np.random.uniform(-theta_widths, theta_widths, (DOF))
    weight = 1
    particles.append(lam)
    weights.append(weight)
weights = np.array(weights) / np.sum(weights)
dist = SetWeightsParticleDistribution(particles, weights, cost, w=[1])

fig = plt.figure()
ax = fig.add_subplot(111)
thetas = [p[0] for p in particles]
ax.hist(thetas, bins=50, normed=True, label='particles')
plt.show()

data = [(2, create_feasible_set(-2, 2))] 
for (theta, feasible) in data:
    feasible = np.ndarray.flatten(feasible)
    dist.weights = dist.reweight(theta, feasible)
    dist.resample()
    particles = dist.particles
    fig = plt.figure()
    ax = fig.add_subplot(111)
    thetas = [p[0] for p in particles]
    ax.hist(thetas, bins=50, normed=True, label='particles')
    ax.hist(feasible, bins=[min(feasible), max(feasible)], normed=True, fc=(0.3, 0.2, 1, 0.5), label='feasible')
    ax.axvline(x=theta, c='C2', label='chosen')
    ax.legend(loc='upper left')
    plt.show()


