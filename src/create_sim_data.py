from __future__ import division
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import probability_estimation as pe

############################################################
# CONSTANTS/FUNCTIONS
DOF = 7
ALPHA = 1
K = 5
lam = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
training_data_size = 1000
f = np.load("./test_results_1.npz")
handle_data = f['human_handoff_ik_solutions'].item()['Mug handle']
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
def get_probability(theta, feasible, lam, cost, ALPHA):
    return pe.prob_theta_given_lam_stable2(theta, lam, feasible, cost, ALPHA)
def get_distribution(feasible, lam, cost, ALPHA):
    return np.array([get_probability(theta, feasible, lam, cost, ALPHA) for theta in feasible])
def create_sample(feasible, probs):
    idx = np.random.choice(len(feasible), p=probs)
    # idx = np.argmax(probs)
    return (feasible[idx], feasible)
def create_sample2(feasible, probs):
    idxs = np.random.choice(len(feasible), p=probs, size=K)
    return (feasible[idxs], feasible)
############################################################
training_data = []
for i in range(training_data_size):
    idx = np.random.choice(handle_data.shape[1])
    # idx = i
    feasible = get_feasible_set(handle_data, idx)
    if feasible is None:
        continue
    probs = get_distribution(feasible, lam, cost, ALPHA)
    training_data.append(create_sample2(feasible, probs))
np.save("sim_k_training_data", training_data)
