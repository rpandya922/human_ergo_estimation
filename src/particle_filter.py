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
DOF = 3
NUM_PARTICLES = 20
h = 0.2
weight_widths = 10
theta_widths = 3.14
num_boxes = 5
axis_ranges = 2 * np.array([weight_widths, weight_widths, weight_widths, theta_widths, theta_widths, theta_widths])
fig = plt.figure()
fig2 = plt.figure()
# plt.ion()
def show_particles(particles, stay=False, time=0.05, entropy='Plot'):
    thetas = np.array(particles)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('w3')
    ax.set_title(entropy)
    ax.set_xlim((-10,10))
    ax.set_ylim((-10,10))
    ax.set_zlim((-10,10))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlabel('theta1')
    ax2.set_ylabel('theta2')
    ax2.set_zlabel('theta3')
    ax2.set_xlim((-3.14,3.14))
    ax2.set_ylim((-3.14,3.14))
    ax2.set_zlim((-3.14,3.14))
    ax2.set_title(entropy)
    if stay:
        print "Staying"
        ax.scatter(thetas[:,0], thetas[:,1], thetas[:,2], c='g', s=10, label='particles', edgecolors='none')
        ax2.scatter(thetas[:,3], thetas[:,4], thetas[:,5], c='g', s=10, label='particles', edgecolors='none')
        plt.legend(loc='upper left')
        while True:
            plt.pause(time)
    else:
        ax.scatter(thetas[:,0], thetas[:,1], thetas[:,2], c='g', s=10, label='particles', edgecolors='none')
        ax2.scatter(thetas[:,3], thetas[:,4], thetas[:,5], c='g', s=10, label='particles', edgecolors='none')
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
            w = particle[:DOF]
            theta_star = particle[DOF:]
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
# feasible_sets = np.load("./feasible_close.npy")
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
    lam = np.random.uniform(-weight_widths, weight_widths, (DOF,))
    lam = np.hstack((lam, np.random.uniform(-theta_widths, theta_widths, (DOF,))))
    weight = 1
    theta = lam[DOF:]
    particles.append(lam)
    weights.append(weight)
weights = np.array(weights) / np.sum(weights)

dist = ParticleDistribution(particles, weights, cost)
dist2 = ParticleDistribution(particles, weights, cost)
show_particles(particles, entropy=dist.entropy(num_boxes, axis_ranges))
expected_info = []
expected_info2 = []
actual_info = []
actual_info2 = []
(x1, theta1, Theta_x1) = data[0]
for (x, theta, Theta_x) in data:
    expected = dist.info_gain(Theta_x, num_boxes, axis_ranges)
    expected_info.append(expected)
    expected2 = dist.info_gain(Theta_x1, num_boxes, axis_ranges)
    expected_info2.append(expected2)
    print "Expected info gain: " + str(expected)
    entropy = dist.entropy(num_boxes, axis_ranges)
    # print entropy
    dist.weights = dist.reweight(theta1, Theta_x1)
    dist2.weights = dist.reweight(theta, Theta_x)
    dist2.resample()
    print "reweighted"
    dist.resample()
    actual = dist.entropy(num_boxes, axis_ranges) - entropy
    actual_info2.append(dist2.entropy(num_boxes, axis_ranges) - entropy)
    dist2 = ParticleDistribution(dist.particles, dist.weights, dist.cost)
    print "Actual info gain: " + str(actual)
    actual_info.append(actual)
    #
    # particles = []
    # weights = []
    # for i in range(NUM_PARTICLES):
    #     lam = np.random.uniform(-weight_widths, weight_widths, (DOF,))
    #     lam = np.hstack((lam, np.random.uniform(-theta_widths, theta_widths, (DOF,))))
    #     weight = 1
    #     theta = lam[DOF:]
    #     particles.append(lam)
    #     weights.append(weight)
    # weights = np.array(weights) / np.sum(weights)
    #
    # dist = ParticleDistribution(particles, weights, cost)
    entropy = dist.entropy(num_boxes, axis_ranges)
    print "Entropy: " + str(entropy)
    print
    show_particles(dist.particles, time=0.05, entropy=entropy)
# show_particles(dist.particles, stay=True, entropy=dist.entropy(num_boxes, axis_ranges))

mean = np.mean(actual_info)
tss = np.sum(np.square(np.array(actual_info) - mean))
rss = np.sum(np.square(np.array(actual_info) - np.array(expected_info2)))
r_squared = 1 - rss / tss
print "R^2 same: "  + str(r_squared)

mean = np.mean(actual_info2)
tss = np.sum(np.square(np.array(actual_info2) - mean))
rss = np.sum(np.square(np.array(actual_info2) - np.array(expected_info)))
r_squared = 1 - rss / tss
print "R^2 new: "  + str(r_squared)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('iteration')
ax.set_ylabel('information gain')
ax.plot(range(len(data)), expected_info, label='expected new sample')
ax.plot(range(len(data)), expected_info2, label='expected same sample')
ax.plot(range(len(data)), actual_info, label='actual same sample')
ax.plot(range(len(data)), actual_info2, label='actual new sample')
plt.legend(loc='upper left')
plt.show()