from __future__ import division
import numpy as np
from scipy.stats import norm
import seaborn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import probability_estimation as pe
from distribution import ParticleDistribution
from scipy.stats import multivariate_normal as mvn
from scipy.stats import gaussian_kde as kde

#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 2
NUM_PARTICLES = 10000
weight_widths = 5
x_widths = 5
num_boxes = 100
axis_ranges = 2 * np.array([weight_widths])
box_size = x_widths * 2 / num_boxes
true_mean = [0, 0]
true_weight = [1, 1]
def cost():
    return 1
def prob(obs, lam):
    if lam[0] < 0 or lam[1] < 0:
        return 0
    cov = np.diag(1 / lam[:DOF])
    mean = lam[DOF:]
    return mvn.pdf(obs, mean=mean, cov=cov)
#########################################################
particles = []
weights = []
for i in range(NUM_PARTICLES):
    lam = np.random.uniform(0, weight_widths, (DOF,))
    lam = np.hstack((lam, np.random.uniform(-x_widths, x_widths, (DOF,))))
    weight = 1
    particles.append(lam)
    weights.append(weight)
weights = np.array(weights) / np.sum(weights)
dist = ParticleDistribution(particles, weights, cost)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
true_cov = np.diag(1 / np.array(true_weight))
for i in range(100):
    print i
    ax1.clear()
    ax2.clear()
    particles = np.array(dist.particles)
    data_weights = particles[:,:2].T
    data_means = particles[:,2:].T
    kernel1 = kde(data_weights)
    kernel2 = kde(data_means)

    xx, yy = np.mgrid[-2:7:100j, -2:7:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel1(positions).T, xx.shape)
    cfset = ax1.contourf(xx, yy, f, cmap='Blues')
    cset = ax1.contour(xx, yy, f, colors='k')
    ax1.clabel(cset, inline=1, fontsize=10)
    ax1.set_title("weights")

    xx, yy = np.mgrid[-6:6:100j, -6:6:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel2(positions).T, xx.shape)
    cfset = ax2.contourf(xx, yy, f, cmap='Blues')
    cset = ax2.contour(xx, yy, f, colors='k')
    ax2.clabel(cset, inline=1, fontsize=10)
    ax2.set_title("mean")
    plt.pause(0.2)

    obs = np.random.multivariate_normal(mean=true_mean, cov=true_cov)
    dist.reweight_general(obs, prob)
    dist.resample()
plt.show()