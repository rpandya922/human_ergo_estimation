from __future__ import division
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../weight_learning')
from openravepy import *
import prpy
import utils
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import probability_estimation as pe
from numpy.random import multivariate_normal as mvn
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde as kde
############################################################
# CONSTANTS/FUNCTIONS
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
seaborn.set_style('ticks')
DOF = 4
ALPHA = 40
FEASIBLE_SIZE = 1000
TRUE_MEAN = np.array([0, 0, 0, 0])
TRUE_WEIGHTS = np.array([4, 3, 2, 1])
training_data_size = 500
FEASIBLE_COLOR = '#42bcf4'
OBJECT = 'lock'
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
    new_data_full = []
    new_data = []
    new_poses = []
    for i in range(data.shape[1]):
        feasible = np.array(get_feasible_set(data, i))
        try:
            if feasible == None:
                continue
        except:
            pass
        if len(feasible) <= 6:
            continue
        # print len(feasible)
        try:
            weights = 1 / kde(feasible.T).evaluate(feasible.T)
        except:
            continue
        weights /= np.sum(weights)
        uniform_feasible_full = feasible[np.random.choice(len(feasible), p=weights,\
        size=len(feasible))]
        # size=min(1000, len(feasible)))]
        uniform_feasible = uniform_feasible_full[:,:4]
        new_data_full.append(uniform_feasible_full)
        new_data.append(uniform_feasible)
        new_poses.append(poses[i])
        print "\r%d" % i,
        sys.stdout.flush()
    print
    return new_data_full, new_data, new_poses
def plot_feas(ax, feasible):
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    ax.set_xlim(-2.3, 0.7)
    ax.set_ylim(-2.5, 0.5)
    ax.scatter(feasible[:,0], feasible[:,1], c=FEASIBLE_COLOR)
    plt.pause(0.01)
##############################################################
# d = np.load('../data/%s_cad_results.npz' % OBJECT)['human_handoff_ik_solutions'].item()
# full_dataset = []
# full_dataset.extend(list(d['Right handle']))
# full_dataset.extend(list(d['Left handle']))
# dataset = np.array(full_dataset)

data = np.load('../data/%s_cad_processed_results.npz' % OBJECT)['data']
poses = np.load('../data/%s_cad_processed_results.npz' % OBJECT)['poses']

env, human, robot, target, target_desc = utils.load_environment_file('../data/%s_cad_problem_def.npz' % OBJECT)
env.SetViewer('qtcoin')
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

newrobots = []
for ind in range(30):
    newrobot = RaveCreateRobot(env,human.GetXMLId())
    newrobot.Clone(human,0)
    for link in newrobot.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(0.8)
    newrobots.append(newrobot)
for link in robot.GetLinks():
    for geom in link.GetGeometries():
        geom.SetTransparency(0.8)
for i in [19]:
    target.SetTransform(poses[i])
    feas_full = data[i]
    with env:
        inds = np.array(np.linspace(0,len(feas_full)-1,30),int)
        for j,ind in enumerate(inds):
            newrobot = newrobots[j]
            env.Add(newrobot,True)
            newrobot.SetTransform(human.GetTransform())
            newrobot.SetDOFValues(feas_full[ind], human.GetActiveManipulator().GetArmIndices())
    env.UpdatePublishedBodies()
    plot_feas(ax, feas_full)
    plt.pause(0.01)
    raw_input('Displaying pose ' + str(i) + ', press <Enter> to continue:')
    ax.cla()