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

handlebars_data = np.load('../data/handlebars_sim_data.npz')['data']
handlebars_poses = np.load('../data/handlebars_sim_data.npz')['poses']
bike_lock_data = np.load('../data/bike_lock_sim_data.npz')['data']
bike_lock_poses = np.load('../data/bike_lock_sim_data.npz')['poses']

# handlebars_feasibles = []
# h_poses = []
# bike_lock_feasibles = []
# bl_poses = []
# for i in [0, 1, 2, 3]:
#     h_feasible = get_feasible_set(handlebars_data, i)
#     handlebars_feasibles.append(h_feasible)
#     h_poses.append(handlebars_poses[i])
# for i in [0, 1, 2, 3]:
#     bl_feasible = get_feasible_set(bike_lock_data, i)
#     bike_lock_feasibles.append(bl_feasible)
#     bl_poses.append(bike_lock_poses[i])
# objects = ['handles', 'handles', 'handles', 'lock', 'lock', 'lock']

handlebars_feasibles = []
h_poses = []
bike_lock_feasibles = []
bl_poses = []
for i in range(len(handlebars_data)):
    handlebars_feasibles.append(handlebars_data[i])
    h_poses.append(handlebars_poses[i])
    bike_lock_feasibles.append(bike_lock_data[i])
    bl_poses.append(bike_lock_poses[i])
objects = ['handles', 'handles', 'handles', 'handles', 'lock', 'lock', 'lock', 'lock']

handlebars_feasibles.extend(bike_lock_feasibles)
all_data = np.array(handlebars_feasibles)
all_poses = np.vstack((h_poses, bl_poses))
all_objects = np.array(objects)

np.savez('../data/handlebars_and_lock_data.npz', data=all_data, poses=all_poses,\
objects=all_objects)
