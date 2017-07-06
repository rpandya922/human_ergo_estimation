from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
import probability_estimation as pe
from distribution import SetWeightsParticleDistribution

#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 1
NUM_PARTICLES = 10000
box_size = 0.05
def cost(theta, theta_star, w):
    return pe.distance_cost(theta, theta_star, w)
def prior(vec):
    return 1
def create_feasible_set(left, right):
    feas = [[left]]
    x = left
    while x <= right:
        if x >= left:
            feas.append([x])
        x += box_size
    feas.append([right])
    return np.array(feas)
#########################################################
particles = []
weights = []
for i in range(NUM_PARTICLES):
    lam = np.random.uniform(-10, 10, (DOF))
    weight = 1
    particles.append(lam)
    weights.append(weight)
weights = np.array(weights) / np.sum(weights)
dist = SetWeightsParticleDistribution(particles, weights, cost, w=[0.1])

fig = plt.figure()
ax = fig.add_subplot(111)
thetas = [p[0] for p in particles]
kernel = kde(thetas)
x = np.linspace(-10, 10, 2000)
pdf = kernel.pdf(x)
mode = x[np.argmax(pdf)]
ax.set_ylim(0, 1.5)
ax.set_xlim(-10, 10)
ax.plot(x, pdf, label='mode = ' + str(mode))
ax.hist(thetas, bins=10, normed=True, label='particles')
ax.legend(loc='upper left')
plt.pause(0.2)

data = [(i - 5, create_feasible_set(-3, 6)) for i in range(10)]
for (theta, feasible) in data:
    dist.weights = dist.reweight(theta, feasible)
    dist.resample()
    feasible = np.ndarray.flatten(feasible)
    particles = dist.particles
    ax.clear()
    thetas = [p[0] for p in particles]
    kernel = kde(thetas)
    pdf = kernel.pdf(x)
    mode = x[np.argmax(pdf)]
    ax.set_ylim(0, 1.5)
    ax.set_xlim(-10, 10)
    ax.plot(x, pdf, label='mode = ' + str(mode))
    ax.hist(thetas, bins=10, normed=True, label='particles')
    ax.hist(feasible, bins=[min(feasible), max(feasible)], normed=True, fc=(0.3, 0.2, 1, 0.5), label='feasible')
    ax.axvline(x=theta, c='C2', label='chosen')
    ax.legend(loc='upper left')
    plt.pause(0.2)
plt.show()


