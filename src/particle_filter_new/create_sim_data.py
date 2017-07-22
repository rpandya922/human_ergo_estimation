from __future__ import division
import sys
sys.path.insert(0, '../')
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import probability_estimation as pe
from numpy.random import multivariate_normal as mvn
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde as kde
############################################################
# CONSTANTS/FUNCTIONS
DOF = 4
ALPHA = 40
FEASIBLE_SIZE = 1000
TRUE_MEAN = np.array([0, 0, 0, 0])
TRUE_WEIGHTS = np.diag([1, 1, 1, 1])
training_data_size = 500

def cost(theta, theta_star, w):
    return np.dot(theta - theta_star, np.dot(w, theta - theta_star))
def get_feasible_set(data, pose):
    sets_list = data[:,pose]
    feasible = []
    for s in sets_list:
        feasible.extend(s)
    feasible = np.array(feasible)
    if len(feasible) == 0:
        return None
    return feasible
def get_distribution(feasible, cost, ALPHA):
    nums = np.array([pe.prob_theta_given_lam_stable_set_weight_num(theta, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA) for theta in feasible])
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feasible, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA)
    return np.exp(nums - denom)
def create_sample(feasible, probs):
    # idx = np.random.choice(len(feasible), p=probs)
    idx = np.argmax(probs)
    return (feasible[idx], feasible)
def preprocess_feasible(data):
    new_data = []
    for i in range(data.shape[1]):
        feasible = np.array(get_feasible_set(data, i))
        if feasible is None or len(feasible) <= 3:
            print "skipped"
            continue
        weights = 1 / kde(feasible.T).evaluate(feasible.T)
        weights /= np.sum(weights)
        uniform_feasible = feasible[np.random.choice(len(feasible), p=weights, size=FEASIBLE_SIZE)][:,:4]
        new_data.append(uniform_feasible)
        if i % 50 == 0:
            print i
    return new_data
##############################################################
data = np.array(preprocess_feasible(np.load('../full_sim_dataset.npy')))
print "preprocessed"
training_data = []
idxs = np.random.choice(len(data), size=training_data_size)
for i in range(training_data_size):
    idx = idxs[i]
    feasible = data[idx]
    probs = get_distribution(feasible, cost, ALPHA)
    training_data.append(create_sample(feasible, probs))
np.save("4joint_sim_training_data_mean0", training_data)
