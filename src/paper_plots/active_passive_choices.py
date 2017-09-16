from __future__ import division
import sys
sys.path.insert(0, '../')
from openravepy import *
import prpy
import numpy as np
import seaborn
seaborn.set_style('ticks')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde as kde
from sklearn.decomposition import PCA
import probability_estimation as pe
from distribution import SetWeightsParticleDistribution
import multiprocessing as mp
from functools import partial
from random import shuffle
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal as mvn
import readline
import argparse
import pickle
import csv
from tqdm import tqdm

def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)

DATA_LOCATION = '../data/user_study'

for subject_idx in tqdm(range(0, 6)):
    try:
        pkl_file = open('%s/subject%s_beleifs.pkl' % (DATA_LOCATION, subject_idx), 'rb')
    except:
        continue
    data = pickle.load(pkl_file)
    expected_costs = data['expected_costs']
    expected_infos = data['expected_infos']

    num_handles_active = 0
    num_lock_active = 0
    for infos in expected_infos:
        m = np.argmax(infos)
        if m <= 3:
            num_handles_active += 1
        else:
            num_lock_active += 1
    num_handles_passive = 0
    num_lock_passive = 0
    for costs in expected_costs:
        m = np.argmin(costs)
        if m <= 3:
            num_handles_passive += 1
        else:
            num_lock_passive += 1
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes = np.ndarray.flatten(np.array(axes))

    axes[0].bar([0, 1], [num_handles_active, num_handles_passive])
    axes[0].set_xticks([0, 1], ('Active', 'Passive'))

    axes[1].bar([0, 1], [num_lock_active, num_lock_passive])
    axes[1].set_xticks([0, 1], ['Active', 'Passive'])
    plt.pause(0.01)
plt.show()