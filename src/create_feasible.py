import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import probability_estimation as pe

data = np.load('./arm_joints_feasible_data.npy')
feasible_sets = np.load("./feasible_sets2.npy")
X, ys = pe.preprocess(data)

far_sets = []
close_sets = []
for i in range(len(X)):
    feasible = np.array(feasible_sets[i])
    p_obj, theta = X[i], pe.normalize(ys[i])
    feasible_sorted = sorted(feasible, key=lambda x: np.linalg.norm(theta - x))
    farthest = feasible_sorted[len(feasible) - 5:]
    closest = feasible_sorted[:5]
    far_sets.append(farthest)
    close_sets.append(closest)
np.save("feasible_far", far_sets)
np.save("feasible_close", close_sets)
