from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import probability_estimation as pe
from numpy.random import multivariate_normal as mvn
from sklearn.neighbors import NearestNeighbors

############################################################
# CONS7ANTS/FUNCTIONS
DOF = 7
ALPHA = 0.5
K = 5
FEASIBLE_SIZE = 5000
lam = np.array([0.5, 0.25, 0.75, 2, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0])
# lam = np.array([1, 0])
training_data_size = 500

def cost(theta, theta_star, w):
    return pe.distance_cost(theta, theta_star, w)
def get_feasible_set():
    feasible = (np.random.rand(40, DOF) * 10) - 5
    return feasible
def get_neighbors_distances(feasible):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(feasible)
    distances, indices = nbrs.kneighbors(feasible)
    return nbrs, np.amax(distances)
def get_distribution(feasible, lam, cost, ALPHA):
    nums = np.array([pe.prob_stable2_num(theta, lam, cost, ALPHA) for theta in feasible])
    denom = pe.prob_stable2_denom(feasible, lam, cost, ALPHA)
    return np.exp(nums - denom)
def create_sample(feasible, probs):
    idxs = list(np.random.choice(len(feasible), p=probs, size=K))
    return (feasible[idxs], feasible)
def preprocess_feasible():
    new_data = []
    for i in range(500):
        feasible = get_feasible_set()
        if feasible is None or len(feasible) <= 3:
            continue
        nbrs, max_dist = get_neighbors_distances(feasible)
        min_bounds = np.amin(feasible, axis=0)
        max_bounds = np.amax(feasible, axis=0)
        new_feas = []
        j = 0
        l = 0
        while l < FEASIBLE_SIZE:
            samples = np.random.uniform(min_bounds, max_bounds, size=(FEASIBLE_SIZE, DOF))
            distances, indices = nbrs.kneighbors(samples)
            for k in range(FEASIBLE_SIZE):
                dist = distances[k][0]
                if dist <= max_dist:
                    new_feas.append(samples[k])
                    l += 1
        new_data.append(np.array(new_feas))
    return new_data
def plot_gamma(alpha, beta):
    scale = 1 / beta
    x = np.linspace(0, 10, 2000)
    y = gamma.pdf(x, a=alpha, scale=scale)
    plt.plot(x, y, label="a: " + str(alpha) + ", beta: " + str(beta))
    plt.title(str((alpha - 1) / beta))
    plt.legend(loc='upper left')
    plt.show()
##############################################################
data = np.array(preprocess_feasible())
print "preprocessed"
training_data = []
idxs = np.random.choice(len(data), size=training_data_size)
for i in range(training_data_size):
    idx = idxs[i]
    feasible = data[idx]
    probs = get_distribution(feasible, lam, cost, ALPHA)
    training_data.append(create_sample(feasible, probs))
np.save("random_training_data_k5", training_data)