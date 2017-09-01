from __future__ import division
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../weight_learning')
import numpy as np
from scipy.stats import gaussian_kde as kde
import probability_estimation as pe
from random import shuffle
import utils
import readline
from scipy.stats import multivariate_normal as mvn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import randn
from scipy import array, newaxis
from mayavi.mlab import *

DEFAULT_DATA_FILE = 'sim_rod_weight_learning.npz'
datafile = utils.prefilled_input('Simulation data file: ', DEFAULT_DATA_FILE)
f = np.load('../data/' + datafile)
data = f['data_full']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for feasible in data:
    ax.cla()
    feasible = feasible[:,:3]
    ax.scatter(feasible[:,0], feasible[:,1], feasible[:,2])
    plt.pause(0.1)
    data_feas = feasible.T
    kernel = kde(data_feas)
    xx, yy, zz = np.mgrid[-3.14:3.14:64j, \
                          -3.14:3.14:64j, \
                          -3.14:3.14:64j]
    positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
    scalars = np.reshape(kernel(positions).T, xx.shape)
    obj = contour3d(xx, yy, zz, scalars)
    show()