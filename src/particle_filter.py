from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import probability_estimation as pe

#########################################################
# CONSTANTS AND FUNCTIONS
NUM_PARTICLES = 1000
h = 0.05
fig = plt.figure()
plt.legend(loc='upper left')
plt.ion()
def show_particles(particles, weights, stay=False):
    thetas = np.array(particles)[:,3:]
    ax = fig.add_subplot(111, projection='3d')
    if stay:
        ax.scatter(thetas[:,0], thetas[:,1], thetas[:,2], c='g', s=100, label='particles')
        while True:
            plt.pause(0.05)
    else:
        ax.scatter(thetas[:,0], thetas[:,1], thetas[:,2], c='g', s=weights*10000, label='particles')
        plt.pause(0.05)
#########################################################

data = np.load('./arm_joints_feasible_data.npy')
feasible_sets = np.load("./feasible_sets2.npy")
X, ys = pe.preprocess(data)
avg = np.mean(ys, axis=0)

data = []
for i in range(len(X)):
    x, y = X[i], pe.normalize(ys[i])
    Y_x = feasible_sets[i]
    Y_x = np.vstack((Y_x, y))
    data.append((x, y, Y_x))

def cost(theta, theta_star, w):
    return pe.distance_cost(theta, theta_star, w)
def prior(vec):
    return 1

particles = []
weights = []
for i in range(NUM_PARTICLES):
    lam = np.random.uniform(-3, 3, (6,))
    weight = 1 / NUM_PARTICLES
    theta = lam[3:]
    particles.append(lam)
    weights.append(weight)

show_particles(particles, weights)

for (x, theta, Theta_x) in data:
    for i in range(NUM_PARTICLES):
        particle = particles[i]
        w = particle[:3]
        theta_star = particle[3:]
        log_likelihood = np.log(weights[i])
        p, costs = pe.prob_theta_given_lam_stable(theta, theta_star, w, Theta_x, cost)
        log_likelihood += p - logsumexp(costs)
        weights[i] = log_likelihood
    weights /= np.sum(weights)
    show_particles(particles, weights)
    new_particles = []
    for i in range(NUM_PARTICLES):
        idx = np.random.choice(NUM_PARTICLES, p=weights)
        sample = particles[idx]
        new_particles.append(np.random.normal(sample, h))
    particles = new_particles
    weights = [1/NUM_PARTICLES for _ in range(NUM_PARTICLES)]

print "Finished"
show_particles(particles, weights, stay=True)