from __future__ import division
import sys
sys.path.insert(0, '../')
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde as kde
import probability_estimation as pe
from distribution import SetWeightsParticleDistribution
import multiprocessing as mp
from functools import partial
from random import shuffle
#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 4
NUM_PARTICLES = 500
box_size = 0.5
ALPHA_I = 0.5
ALPHA_O = 0.1
TRUE_MEAN = [0, 0]
def cost(theta, theta_star, w):
    return np.dot(theta - theta_star, np.dot(w, theta - theta_star))
#########################################################
particles = []
weights = []
for i in range(NUM_PARTICLES):
    lam = np.random.uniform(-10, 10, (DOF))
    weight = 1
    particles.append(lam)
    weights.append(weight)
weights = np.array(weights) / np.sum(weights)
w = np.diag([0.5, 0.5, 0.5, 0.5])
dist = SetWeightsParticleDistribution(particles, weights, cost, w=w, ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)

