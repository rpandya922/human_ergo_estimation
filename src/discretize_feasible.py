from __future__ import division
import numpy as np
import math
import decimal
from random import shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize(angles):
    s = []
    for x in angles:
        while x < -3.14:
            x += 6.28
        while x > 3.14:
            x -= 6.28
        s.append(x)
    return np.array(s)

data = np.load("./arm_joints_feasible_data.npy")
box_size = 6.28 / 100
num_boxes = (6.28 / box_size) ** 3
feasible_sets = []
for d in data:
    v = d[1][:3]
    traj = np.array(np.array(d[2])[:,:3])
    theta = set()
    for sample in traj:
        s = normalize(sample)
        s = ((s // box_size) * box_size) + (box_size / 2)
        theta.add(tuple(s))
    feasible_sets.append(np.array(list(theta)))
    f = np.array(list(theta))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=f[:,0], ys=f[:,1], zs=f[:,2])
    ax.scatter(v[0], v[1], v[2], c='r', marker='^')
    plt.show()
np.save("./feasible_sets2", feasible_sets)

# x = np.array(data[5][2])[:,:3]
# size = len(x[0])
# for i in range(len(x)):
#     for j in range(size):
#         if x[i][j] < 0:
#             x[i][j] += 2 * math.pi
# print x
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim([-3.14, 3.14])
# ax.set_ylim([-3.14, 3.14])
# ax.set_zlim([-3.14, 3.14])
#
# ax.scatter(xs=x[:,0], ys=x[:,1], zs=x[:,2])
# plt.show()