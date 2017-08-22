from __future__ import division
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../weight_learning')
import numpy as np
import seaborn
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
import probability_estimation as pe
from random import shuffle
import utils
import readline
from scipy.stats import multivariate_normal as mvn
import matplotlib
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from numpy.random import randn
from scipy import array, newaxis

DEFAULT_DATA_FILE = 'sim_rod_weight_learning.npz'
datafile = utils.prefilled_input('Simulation data file: ', DEFAULT_DATA_FILE)
f = np.load('../data/' + datafile)
idxs = list(range(25))
data = f['data']

for theta, feasible in data:
    data_feas = feasible.T
    kernel = kde(data_feas)
    new_feas = kernel.resample(20).T

    Xs = new_feas[:,0]
    Ys = new_feas[:,1]
    Zs = new_feas[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(Xs, Ys, Zs, cmap='Blues', linewidth=0)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()
    plt.show()