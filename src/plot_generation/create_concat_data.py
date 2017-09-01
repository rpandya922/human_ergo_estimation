import numpy as np
from random import sample

def get_feasible_set(data, pose):
    sets_list = data[:,pose]
    feasible = []
    for s in sets_list:
        feasible.extend(s)
    feasible = np.array(feasible)
    if len(feasible) == 0:
        return None
    return feasible

rod_data = np.load('../data/sim_data_rod.npy')
rod_poses = np.load('../data/rod_full_cases.npz')['pose_samples']
objects = ['rod' for _ in range(len(rod_data[1]))]
mug_data = np.load('../data/sim_data_translations_varied.npy')
mug_poses = np.load('../data/test_cases_varied.npz')['pose_samples']
mug_objects = ['mug' for _ in range(len(mug_data[1]))]

rod_feasibles = []
mug_feasibles = []
for i in range(rod_data.shape[1]):
    r_feasible = get_feasible_set(rod_data, i)
    m_feasible = get_feasible_set(mug_data, i)
    rod_feasibles.append(r_feasible)
    mug_feasibles.append(m_feasible)
rod_feasibles.extend(mug_feasibles)
objects.extend(mug_objects)

all_data = np.array(rod_feasibles)
all_poses = np.vstack((rod_poses, mug_poses))
all_objects = np.array(objects)

idxs = sample(list(range(len(all_data))), len(all_data))
shuffled_data = all_data[idxs]
shuffled_poses = all_poses[idxs]
shuffled_objects = all_objects[idxs]

np.savez('../data/rod_and_mug_data.npz', data=shuffled_data, poses=shuffled_poses,\
objects=shuffled_objects)
