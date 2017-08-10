from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
import probability_estimation as pe
from distribution import SetWeightsParticleDistribution
import multiprocessing as mp
from functools import partial

#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 1
NUM_PARTICLES = 1000
box_size = 0.1
ALPHA_I = 1
ALPHA_O = 1
TRUE_MEAN = 0
def cost(theta, theta_star, w):
    return pe.distance_cost1d(theta, theta_star, w)
def prior(vec):
    return 1
def create_feasible_set(left, right, chosen):
    feas = [[left]]
    x = left
    while x <= right:
        if x >= left:
            feas.append([x])
        x += box_size
    feas.append([right])
    feas.append([chosen])
    return np.unique(feas, axis=0)
#########################################################
particles = []
weights = []
for i in range(NUM_PARTICLES):
    lam = np.random.uniform(-10, 10, (DOF))
    weight = 1
    particles.append(lam)
    weights.append(weight)
weights = np.array(weights) / np.sum(weights)
dist = SetWeightsParticleDistribution(particles, weights, cost, w=[0.5], ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
axes = np.ndarray.flatten(np.array(axes))

ax = axes[0]
thetas = [p[0] for p in particles]
kernel = kde(thetas)
x = np.linspace(-12, 12, 2000)
pdf = kernel.pdf(x)
mode = x[np.argmax(pdf)]
ax.set_ylim(0, 1.5)
ax.set_xlim(-12, 12)
ax.plot(x, pdf, label='kde, mode = ' + str(mode))
ax.hist(thetas, bins=10, normed=True, label='particles')
ax.axvline(x=TRUE_MEAN, c='C3', label='true mean')
ax.set_title('initialization')
ax.legend(loc='upper left')
plt.pause(0.2)

# data = [(0, create_feasible_set(-10, 0, 0)), (3, create_feasible_set(3, 8, 3)), \
#         (-1, create_feasible_set(-4, -1, -1)), (4, create_feasible_set(4, 7, 4)), \
#         (0, create_feasible_set(-2, 2, 0))]
# data = [(0, create_feasible_set(-2, 2, 0))] * 5
data = [(0, create_feasible_set(0, 3, 0)), (1, create_feasible_set(1, 1, 5)), \
        (2, create_feasible_set(2, 4, 2))] * 2
for i in range(1, 6):
    ax = axes[i]
    ax_prev = axes[i-1]
    (theta, feasible) = data[i-1]

    dist.weights = dist.reweight(theta, feasible)
    dist.resample()

    feasible = np.ndarray.flatten(feasible)
    particles = dist.particles
    thetas = [p[0] for p in particles]
    kernel = kde(thetas)
    pdf = kernel.pdf(x)
    mode = x[np.argmax(pdf)]
    ax.set_ylim(0, 0.5)
    ax.set_xlim(-12, 12)
    ax.plot(x, pdf, label='mode = ' + str(mode))
    ax.hist(thetas, bins=10, normed=True, label='particles')
    # ax.hist(feasible, bins=[min(feasible), max(feasible)], normed=True, fc=(0.3, 0.2, 1, 0.5), label='feasible')
    # ax.axvline(x=theta, c='C2', label='chosen')
    ax.axvline(x=TRUE_MEAN, c='C3', label='true mean')
    ax.legend(loc='upper left')
    # ax_prev.hist(feasible, bins=[min(feasible), max(feasible)], normed=True, fc=(0.37, 0.11, 0.51, 0.5), label='feasible set')
    # ax_prev.axvline(x=theta, c='C2', label='chosen theta')
    ax.hist(feasible, bins=[min(feasible), max(feasible)], normed=True, fc=(0.3, 0.2, 1, 0.5), label='feasible set')
    ax.axvline(x=theta, c='C2', label='chosen theta')
    ax.set_title('iteration' + str(i))
    ax_prev.legend(loc='upper left')
    plt.pause(0.2)
plt.show()
