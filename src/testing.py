from __future__ import division
import numpy as np
import probability_estimation as pe
##################################################
# CONSTANTS
DOF = 7
samples = 1000
lam = np.array([5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0])
# lam = np.array([5, 5, 0, 0])
K = 5
ALPHA = 0.5
def cost(theta, theta_star, w):
    return pe.distance_cost(theta, theta_star, w)
def get_feasible_set():
    feas_size = 100
    feasible = (np.random.rand(feas_size, DOF) * 2) - 1
    return feasible
def get_probability(theta, feasible, lam, cost, ALPHA):
    return pe.prob_theta_given_lam_stable2(theta, lam, feasible, cost, ALPHA)
def get_distribution(feasible, lam, cost, ALPHA):
    return np.array([get_probability(theta, feasible, lam, cost, ALPHA) for theta in feasible])
def create_sample2(feasible, probs):
    idxs = np.random.choice(len(feasible), p=probs, size=K)
    return (feasible[idxs], feasible)
##################################################
training_data = []
for i in range(samples):
    feasible = get_feasible_set()
    probs = get_distribution(feasible, lam, cost, ALPHA)
    training_data.append(create_sample2(feasible, probs))
np.save("random_training_data", training_data)