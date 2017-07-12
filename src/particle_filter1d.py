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
NUM_PARTICLES = 200
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
dist = SetWeightsParticleDistribution(particles, weights, cost, w=[1], ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)
# dist.weights = dist.reweight(-8, create_feasible_set(-10, -8, -8))
# dist.resample()
# particles = dist.particles

def info_gain(dist, x):
    return (x, dist.info_gain(x[1], num_boxes=20))
if __name__ == '__main__':
    pool = mp.Pool(6)
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)
    axes = np.ndarray.flatten(np.array(axes))
    ax = axes[0]
    thetas = [p[0] for p in particles]
    kernel = kde(thetas)
    x = np.linspace(-12, 12, 2000)
    pdf = kernel.pdf(x)
    mode = x[np.argmax(pdf)]
    ax.set_ylim(0, 1.5)
    ax.set_xlim(-12, 12)
    ax.plot(x, pdf, label='mode = ' + str(mode))
    ax.hist(thetas, bins=10, normed=True, label='particles')
    ax.axvline(x=TRUE_MEAN, c='C3', label='true mean')
    ax.legend(loc='upper left')
    plt.pause(0.2)
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)

    exp_infos = []
    actual_infos = []

    # data = [(6, create_feasible_set(6, 9, 6)), (2, create_feasible_set(-4, 2, 2)), \
    #         (2, create_feasible_set(-2, 2, 2)), (5.5, create_feasible_set(2, 5.5, 5.5)), \
    #         (5.5, create_feasible_set(2.1, 5.6, 5.5)), (-2, create_feasible_set(-8, -2, -2))]
    data = [(-1, create_feasible_set(-4, -1, -1)), (1, create_feasible_set(1, 4, 1)), \
            (4, create_feasible_set(4, 6, 4)), (6, create_feasible_set(6, 10, 6)), \
            (0, create_feasible_set(-1, 1, 0)), (-4, create_feasible_set(-8, -4, -4))]
    # data = [(6, create_feasible_set(6, 9, 6)), (4, create_feasible_set(2, 6, 4)), (2, create_feasible_set(-2, 2, 2))]
    # data = [(-2, create_feasible_set(-4, -2, -2)), (2, create_feasible_set(2, 4, 2))]
    for i in range(1, 32):
        ax = axes[(i % 16)]
        ax.clear()
        ax_prev = axes[((i-1) % 16)]
        func = partial(info_gain, dist)
        (theta, feasible) = max(pool.map(func, data), key=lambda x: x[1])[0]
        # exp_infos.append(dist.info_gain(feasible, num_boxes=20))
        # entr = dist.entropy(num_boxes=20)
        print
        # (theta, feasible) = max(data, key=lambda x: dist.info_gain(x[1]))
    # for (theta, feasible) in data:
        dist.weights = dist.reweight(theta, feasible)
        dist.resample()
        # actual_infos.append(entr - dist.entropy(num_boxes=20))
        feasible = np.ndarray.flatten(feasible)
        particles = dist.particles
        thetas = [p[0] for p in particles]
        kernel = kde(thetas)
        pdf = kernel.pdf(x)
        mode = x[np.argmax(pdf)]
        ax.set_ylim(0, 1.5)
        ax.set_xlim(-12, 12)
        ax.plot(x, pdf, label='mode = ' + str(mode))
        ax.hist(thetas, bins=10, normed=True, label='particles')
        ax.hist(feasible, bins=[min(feasible), max(feasible)], normed=True, fc=(0.3, 0.2, 1, 0.5), label='feasible')
        ax.axvline(x=theta, c='C2', label='chosen')
        ax.axvline(x=TRUE_MEAN, c='C3', label='true mean')
        ax.legend(loc='upper left')

        ax_prev.hist(feasible, bins=[min(feasible), max(feasible)], normed=True, fc=(0.37, 0.11, 0.51, 0.5), label='next max info feas')
        ax_prev.legend(loc='upper left')
        plt.pause(0.2)
    # ax2.set_xlabel('iteration')
    # ax2.set_ylabel('information gain')
    # ax2.plot(exp_infos, label='expected')
    # ax2.plot(actual_infos, label='actual')
    # ax2.legend(loc='upper left')
    plt.show()
