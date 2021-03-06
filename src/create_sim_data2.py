from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import probability_estimation as pe
from numpy.random import multivariate_normal as mvn
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde as kde
############################################################
# CONSTANTS/FUNCTIONS
DOF = 7
ALPHA = 0.5
K = 2
FEASIBLE_SIZE = 5000
lam = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
training_data_size = 500
# f = np.load("./test_results_1.npz")
# handle_data = f['human_handoff_ik_solutions'].item()['Mug handle']

def cost(theta, theta_star, w):
    return pe.distance_cost(theta, theta_star, w)
def get_feasible_set(data, pose):
    sets_list = data[:,pose]
    feasible = []
    for s in sets_list:
        feasible.extend(s)
    feasible = np.array(feasible)
    if len(feasible) == 0:
        return None
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
def preprocess_feasible(data):
    new_data = []
    for i in range(data.shape[1]):
        feasible = get_feasible_set(data, i)
        if feasible is None or len(feasible) <= 3:
            print "skipped"
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
            for k in range(len(samples)):
                dist = distances[k][0]
                if dist <= max_dist:
                    new_feas.append(samples[k])
                    l += 1
        #     j += 1
        # print j
        if i % 50 == 0:
            print i
        new_data.append(np.array(new_feas))
    return new_data
def preprocess_feasible2(data):
    new_data = []
    for i in range(data.shape[1]):
        feasible = np.array(get_feasible_set(data, i))
        if feasible is None or len(feasible) <= 3:
            print "skipped"
            continue
        weights = 1 / kde(feasible.T).evaluate(feasible.T)
        weights /= np.sum(weights)
        uniform_feasible = feasible[np.random.choice(len(feasible), p=weights, size=FEASIBLE_SIZE)]
        new_data.append(uniform_feasible)
        if i % 50 == 0:
            print i
    return new_data
##############################################################
# data = np.array(preprocess_feasible(handle_data))
data = np.array(preprocess_feasible2(np.load('./full_sim_dataset.npy')))
print "preprocessed"
training_data = []
idxs = np.random.choice(len(data), size=training_data_size)
for i in range(training_data_size):
    idx = idxs[i]
    feasible = data[idx]
    probs = get_distribution(feasible, lam, cost, ALPHA)
    training_data.append(create_sample(feasible, probs))
np.save("full_sim_k_training_data_mean0_k2", training_data)
