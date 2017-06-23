from __future__ import division
import seaborn
import numpy as np
from scipy.optimize import minimize
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
import probability_estimation as pe
from scipy.optimize import minimize
from random import shuffle

#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 4
true_lam = np.array([1, 1, 1, 1, 1, 1, 1, 1])
def get_probability(theta, feasible, lam, cost, ALPHA):
    return pe.prob_theta_given_lam_stable2(theta, lam, feasible, cost, ALPHA)
def get_distribution(feasible, lam, cost, ALPHA):
    return np.array([get_probability(theta, feasible, lam, cost, ALPHA) for theta in feasible])
#########################################################
data = np.load('./sim_training_data.npy')[:70]
test_data = data[:25]
data = data[25:]

print len(data)
def cost(theta, theta_star, w):
    return pe.distance_cost(theta, theta_star, w)
def prior(vec):
    return 1
def objective(lam):
    return -np.sum([pe.log_likelihood_theta_given_lam(theta, lam, feasible, cost) for (theta, feasible) in data])
res = minimize(objective, [0, 0, 0, 0, 0, 0, 0, 0], options={'disp': True})
lam = res.x
print lam

print -np.sum([pe.log_likelihood_theta_given_lam(theta, true_lam, feasible, cost) for (theta, feasible) in test_data])
print -np.sum([pe.log_likelihood_theta_given_lam(theta, lam, feasible, cost) for (theta, feasible) in test_data])


# expected = []
# calculated = []
# for (_, feasible) in test_data:
#     for theta in feasible:
#         expect = pe.prob_theta_given_lam_stable2(theta, true_lam, feasible, cost, 1)
#         calculate = pe.prob_theta_given_lam_stable2(theta, lam, feasible, cost, 1)
#         expected.append(expect)
#         calculated.append(calculate)
#     # print "Expected Prob: " + str(expected)
#     # print "Calculated Prob: " + str(calculated)
#     # print
# mean = np.mean(expected)
# tss = np.sum(np.square(np.array(expected) - mean))
# rss = np.sum(np.square(np.array(calculated) - np.array(expected)))
# r_squared = 1 - rss / tss
# print "R^2: "  + str(r_squared)