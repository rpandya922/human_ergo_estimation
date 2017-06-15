from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import probability_estimation as pe
from distribution import ParticleDistribution

#########################################################
# CONSTANTS AND FUNCTIONS
NUM_PARTICLES = 10000
h = 0.2
fig = plt.figure()
# plt.ion()
def show_particles(particles, stay=False, time=0.05):
    thetas = np.array(particles)[:,3:]
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if stay:
        print "Staying"
        ax.scatter(thetas[:,0], thetas[:,1], thetas[:,2], c='g', s=10, label='particles', edgecolors='none')
        plt.legend(loc='upper left')
        while True:
            plt.pause(time)
    else:
        ax.scatter(thetas[:,0], thetas[:,1], thetas[:,2], c='g', s=10, label='particles', edgecolors='none')
        plt.legend(loc='upper left')
        plt.pause(time)
def compare(particles):
    def objective_stable(lam):
        return -np.sum([pe.prob_lam_given_theta_stable(theta, lam, Theta_x, cost, prior) for (x, theta, Theta_x) in full_data])
    res = minimize(objective_stable, [0, 1, 0, 0, 0, 0], method='Powell', options={'disp': True})
    l = res.x
    l2 = min(particles, key=lambda x: objective_stable(x))
    print "Optimized"
    pe.evaluate_lambda(l, full_data, cost, prior)
    print "Min Particles"
    pe.evaluate_lambda(l2, full_data, cost, prior)
def create_feasible_set(theta, stddev, size):
    f = [np.random.normal(0, stddev, theta.shape) + theta for _ in range(size)]
    f.append(y)
    return np.array(f)
def filter_with_variance(particles, weights):
    variances = []
    indices = []
    k = 0
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
        print "reweighted"
        new_particles = []
        idxs = np.random.choice(NUM_PARTICLES, size=NUM_PARTICLES, p=weights)
        for i in range(NUM_PARTICLES):
            idx = idxs[i]
            sample = particles[idx]
            new_particles.append(np.random.normal(sample, h))
        particles = new_particles
        weights = [1/NUM_PARTICLES]*NUM_PARTICLES
        var = np.var(particles, axis=0)
        variances.append(var)
        indices.append(k)
        k += 1
    variances = np.array(variances)
    feasible_sizes = np.array(feasible_sizes)
    for i in range(6):
        plt.xlabel('iteration')
        plt.ylabel('variance')
        plt.plot(indices, variances[:,i])
        plt.show()
        plt.savefig("/home/ravi/figures/feasible_vars/feas_far_var" + str(i) + ".png")
        plt.clf()
#########################################################

data = np.load('./arm_joints_feasible_data.npy')
# full_feasible_sets = np.load("./feasible_sets2.npy")
# feasible_sets = np.load("./feasible_far.npy")
feasible_sets = np.load("./feasible_sets2.npy")
X, ys = pe.preprocess(data)
avg = np.mean(ys, axis=0)

# full_data = []
# for i in range(len(X)):
#     x, y = X[i], pe.normalize(ys[i])
#     Y_x = full_feasible_sets[i]
#     Y_x = np.vstack((Y_x, y))
#     full_data.append((x, y, Y_x))

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
    lam = np.random.uniform(-10, 10, (6,))
    weight = 1 / NUM_PARTICLES
    theta = lam[3:]
    particles.append(lam)
    weights.append(weight)

show_particles(particles)
dist = ParticleDistribution(particles, cost)
for (x, theta, Theta_x) in data:
    dist.resample(theta, Theta_x)
    print dist.entropy(box_size=5)
    show_particles(dist.particles)
show_particles(dist.particles, stay=True)
