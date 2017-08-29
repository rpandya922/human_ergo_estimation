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
from matplotlib import cm
from numpy.random import randn
from scipy import array, newaxis
from mayavi.mlab import *

DEFAULT_DATA_FILE = 'sim_rod_weight_learning.npz'
datafile = utils.prefilled_input('Simulation data file: ', DEFAULT_DATA_FILE)
f = np.load('../data/' + datafile)
idxs = list(range(25))
data = f['data']

theta, feasible = data[0]
theta = theta[:3]
feasible = feasible[:,:3]
data_feas = feasible.T
kernel = kde(data_feas)
x, y, z = np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]
scalars = kernel(np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j])
print scalars.shape
1/0
scalars = x * x * 0.5 + y * y + z * z * 2.0
obj = contour3d(scalars, contours=4, transparent=True)
show()