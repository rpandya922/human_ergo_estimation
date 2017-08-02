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
TRUE_WEIGHTS = np.array([1, 1, 1, 1])
training_data_size = 500

def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)
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
def preprocess_feasible(data, poses):
    new_data = []
    new_poses = []
    for i in range(data.shape[1]):
        feasible = np.array(get_feasible_set(data, i))
        try:
            if feasible == None:
                continue
        except:
            pass
        if len(feasible) <= 3:
            continue
        weights = 1 / kde(feasible.T).evaluate(feasible.T)
        weights /= np.sum(weights)
        uniform_feasible = feasible[np.random.choice(len(feasible), p=weights, size=len(feasible))][:,:4]
        new_data.append(uniform_feasible)
        new_poses.append(poses[i])
        print "\r%d" % i,
        sys.stdout.flush()
    print
    return new_data, new_poses
##############################################################
data, poses = np.array(preprocess_feasible(np.load('sim_data_translations.npy'), np.load('test_cases.npz')['pose_samples']))
print "preprocessed"
training_data = []
for i in range(len(data)):
    feasible = data[i]
    probs = get_distribution(feasible, cost, ALPHA)
    training_data.append(create_sample(feasible, probs))
np.savez("sim_translation_training_data_diff_sizes", data=training_data, poses=poses)
