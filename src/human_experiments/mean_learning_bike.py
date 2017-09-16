from __future__ import division
import sys
sys.path.insert(0, '../')
from openravepy import *
import prpy
import numpy as np
import seaborn
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
import readline
import argparse
import pickle
from tqdm import tqdm
from scipy.stats import multivariate_normal as mvn

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=-1)
parser.add_argument('--sets', nargs='+', type=int, default=list(range(20)))
args = parser.parse_args()
if args.subject == -1:
    print "subject argument required"
    sys.exit()
################################################################################
# CONSTANTS/FUNCTIONS
DOF = 7
ALPHA = 40
ALPHA_I = 1
ALPHA_O = 1
RESAMPLING_VARIANCE = 0.1
FEASIBLE_SIZE = 1000
TRUE_MEAN = np.array([0, 0, 0, 1.57, 0, 0, 0])
TRUE_WEIGHTS = np.array([1, 1, 1, 1, 1, 1, 1])
NUM_PARTICLES = 1000
NUM_TRAIN_ITERATIONS = 6
training_data_size = 500
DISTRIBUTION_DATA_FOLDER = '../data/user_study'
TSR_SPLITS = [44, 133, 24, 14, 128, 110, 33, 266]
CONFIG_NAMES = ['handles_vertical', 'handles_vertical_inverse_skew', 'handles_perp', \
'handles_vertical_inverse', 'lock_vertical', 'lock_vertical_skew', 'lock_vertical_inverse',\
'lock_horizontal']
def load_object_desc(desc_string):
    result = eval(desc_string)
    for tsr_name in result['human_tsrs'].keys():
        tsr_obj = prpy.tsr.TSR(**result['human_tsrs'][tsr_name])
        result['human_tsrs'][tsr_name] = tsr_obj
    for tsr_name in result['robot_tsrs'].keys():
        tsr_obj = prpy.tsr.TSR(**result['robot_tsrs'][tsr_name])
        result['robot_tsrs'][tsr_name] = tsr_obj
    return result
def load(filename):
    return np.load(filename)
def load_txt(filename):
    with open(filename, 'r') as file_handle:
        object_string = file_handle.read()
    return load_object_desc(object_string)
def load_environment_file(filename):
    problem_def = load(filename)
    human_file = problem_def['human_file'].tostring()
    robot_file = problem_def['robot_file'].tostring()
    object_file = problem_def['object_file'].tostring()
    target_desc = load_txt('../data/' + object_file)
    human_base_pose = problem_def['human_base_pose']
    robot_base_pose = problem_def['robot_base_pose']
    object_start_pose = problem_def['object_start_pose']
    problem_def.close()
    return load_environment(human_file, robot_file, object_file,
            human_base_pose, robot_base_pose, object_start_pose)
def load_environment(human_file, robot_file, object_file,
        human_base_pose, robot_base_pose, object_start_pose):
    env = Environment()

    #Add the human
    human = env.ReadKinBodyXMLFile(human_file)
    env.AddKinBody(human)

    env.GetCollisionChecker().SetCollisionOptions(0)
    manip = human.SetActiveManipulator('rightarm')
    human.SetTransform(human_base_pose)

    human.GetLink('Hips').SetVisible(False)
    hand_joints = []
    hand_joints.append(human.GetJointIndex('JLFing11'))
    hand_joints.append(human.GetJointIndex('JLFing21'))
    hand_joints.append(human.GetJointIndex('JLFing31'))
    hand_joints.append(human.GetJointIndex('JLFing41'))
    hand_joints.append(human.GetJointIndex('JLFing10'))
    hand_joints.append(human.GetJointIndex('JLFing20'))
    hand_joints.append(human.GetJointIndex('JLFing30'))
    hand_joints.append(human.GetJointIndex('JLFing40'))
    hand_joints.append(human.GetJointIndex('JRFing11'))
    hand_joints.append(human.GetJointIndex('JRFing21'))
    hand_joints.append(human.GetJointIndex('JRFing31'))
    hand_joints.append(human.GetJointIndex('JRFing41'))
    hand_joints.append(human.GetJointIndex('JRFing10'))
    hand_joints.append(human.GetJointIndex('JRFing20'))
    hand_joints.append(human.GetJointIndex('JRFing30'))
    hand_joints.append(human.GetJointIndex('JRFing40'))
    human.SetDOFValues([0.5]*16, hand_joints)

    #Add the robot
    with env:
         robot = env.ReadRobotXMLFile(robot_file)
    #Add the object
    target_desc = load_txt('../data/' + object_file)
    with env:
        target = env.ReadKinBodyXMLFile(target_desc['object_file'])
        print target_desc['object_file']
        env.AddKinBody(target)
        target.SetTransform(object_start_pose)

    return env, human, robot, target, target_desc
def plot_pose(feasible, pose):
    target.SetTransform(pose)
    with env:
        inds = np.array(np.linspace(0,len(feasible)-1,15),int)
        for j,ind in enumerate(inds):
            newrobot = newrobots[j]
            env.Add(newrobot,True)
            newrobot.SetTransform(human.GetTransform())
            newrobot.SetDOFValues(feasible[ind], human.GetActiveManipulator().GetArmIndices())
    env.UpdatePublishedBodies()
    # raw_input('Displaying pose ' + str(i) + ', press <Enter> to continue:')
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
def get_distribution(feasible, cost, ground_truth, ALPHA):
    nums = np.array([pe.prob_theta_given_lam_stable_set_weight_num(theta, ground_truth, TRUE_WEIGHTS, cost, ALPHA) for theta in feasible])
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feasible, ground_truth, TRUE_WEIGHTS, cost, ALPHA)
    return np.exp(nums - denom)
def create_sample(feasible, probs):
    idx = np.argmax(probs)
    return (feasible[idx], feasible)
def preprocess_feasible(data, poses, get_feasible=True):
    new_data_full = []
    new_poses = []
    if get_feasible:
        num_iterations = data.shape[1]
    else:
        num_iterations = len(data)
    for i in tqdm(range(num_iterations)):
    # for i in tqdm(range(70)):
        if get_feasible:
            feasible = np.array(get_feasible_set(data, i))
        else:
            feasible = np.array(data[i])
        try:
            if feasible == None:
                continue
        except:
            pass
        if len(feasible) <= 2:
            continue
        new_data_full.append(feasible)
        new_poses.append(poses[i])
    return new_data_full, new_poses
def prob_of_truth(dist, ground_truth):
    DOF = len(dist.particles[0])
    cov = np.diag(np.ones(DOF)) * 0.0625
    likelihoods = mvn.pdf(dist.particles, mean=ground_truth, cov=cov)
    return np.sum(likelihoods) / dist.NUM_PARTICLES
def dist_to_truth(dist, ground_truth):
    mode = dist.distribution_mode()
    return np.linalg.norm(mode - ground_truth)
def get_theta(dist, feasible, tsr_split):
    tsr = raw_input("Select which TSR was chosen (1/2): ")
    while tsr != '1' and tsr != '2':
        tsr = raw_input("%s is not a valid TSR. Select 1 or 2: " % tsr)
    if tsr == '1':
        tsr_feas = feasible[:tsr_split]
        dist_mode = dist.distribution_mode()
        probs = get_distribution(tsr_feas, cost, dist_mode, ALPHA)
        theta, _ = create_sample(tsr_feas, probs)
    elif tsr == '2':
        tsr_feas = feasible[tsr_split:]
        dist_mode = dist.distribution_mode()
        probs = get_distribution(tsr_feas, cost, dist_mode, ALPHA)
        theta, _ = create_sample(tsr_feas, probs)
    return theta
def get_theta_ground_truth(ground_truth, feasible, tsr_split):
    tsr = raw_input("Select which TSR was chosen (1/2): ")
    while tsr != '1' and tsr != '2':
        tsr = raw_input("%s is not a valid TSR. Select 1 or 2: " % tsr)
    if tsr == '1':
        tsr_feas = feasible[:tsr_split]
        probs = get_distribution(tsr_feas, cost, ground_truth, ALPHA)
        theta, _ = create_sample(tsr_feas, probs)
    elif tsr == '2':
        tsr_feas = feasible[tsr_split:]
        probs = get_distribution(tsr_feas, cost, ground_truth, ALPHA)
        theta, _ = create_sample(tsr_feas, probs)
    return theta
def train_active(dist, data, poses):
    all_particles = [np.copy(dist.particles)]
    all_expected_infos = []
    for i in range(1, NUM_TRAIN_ITERATIONS):
        print "\rActive on iteration %d of 5. " % i,
        sys.stdout.flush()
        func = partial(info_gain, dist)
        pooled = pool.map(func, data)
        expected_infos = [sample[1] for sample in pooled]
        max_idx = np.argmax(expected_infos)
        tsr_split = TSR_SPLITS[max_idx]
        feasible = pooled[max_idx][0]
        print "Configuration %s selected, index %s" % (CONFIG_NAMES[max_idx], max_idx)
        plot_pose(feasible, poses[max_idx])
        theta = get_theta(dist, feasible, tsr_split)
        actual_infos = []
        ent_before = dist.entropy(num_boxes=20)

        dist.weights = dist.reweight(theta, feasible)
        dist.resample()

        all_expected_infos.append(expected_infos[:])
        all_particles.append(np.copy(dist.particles))
    print
    return all_expected_infos, all_particles
def train_min_cost(dist, data, poses):
    all_particles = [np.copy(dist.particles)]
    all_expected_costs = []
    for i in range(1, NUM_TRAIN_ITERATIONS):
        print "\rPassive on iteration %d of 5. " % i,
        sys.stdout.flush()
        func = partial(min_cost, dist)
        pooled = pool.map(func, data)
        expected_costs = [sample[2] for sample in pooled]
        max_idx = np.argmin(expected_costs)
        tsr_split = TSR_SPLITS[max_idx]
        feasible = pooled[max_idx][0]
        print "Configuration %s selected, index %s" % (CONFIG_NAMES[max_idx], max_idx)
        plot_pose(feasible, poses[max_idx])
        theta = get_theta(dist, feasible, tsr_split)

        dist.weights = dist.reweight(theta, feasible)
        dist.resample()

        all_expected_costs.append(expected_costs[:])
        all_particles.append(np.copy(dist.particles))
    print
    return all_expected_costs, all_particles
def train_random(dist, data, poses):
    all_particles = [np.copy(dist.particles)]
    for i in range(1, NUM_TRAIN_ITERATIONS):
        print "\rRandom on iteration %d of 5. " % i,
        sys.stdout.flush()
        idx = np.random.choice(len(data))
        tsr_split = TSR_SPLITS[idx]
        feasible = data[idx]
        print "Configuration %s selected, index %s" % (CONFIG_NAMES[idx], idx)
        plot_pose(feasible, poses[idx])
        theta = get_theta(dist, feasible, tsr_split)

        dist.weights = dist.reweight(theta, feasible)
        dist.resample()

        all_particles.append(np.copy(dist.particles))
    print
    return all_particles
def collect_training_set(data, poses):
    training_set = []
    for i, feasible in enumerate(data):
        print i
        pose = poses[i]
        tsr_split = TSR_SPLITS[i]
        plot_pose(feasible, pose)
        theta = get_theta_ground_truth(TRUE_MEAN, feasible, tsr_split)
        training_set.append((np.copy(theta), np.copy(feasible)))
    np.savez("../data/user_study/subject%s_training_set.npz" % args.subject, training_set=training_set)
def collect_test_set(handles_data, handles_poses, lock_data, lock_poses):
    handles_tsr_splits = np.load('../data/handlebars_tsr_splits.npz')['tsr_splits']
    lock_tsr_splits = np.load('../data/bike_lock_tsr_splits.npz')['tsr_splits']
    idxs = list(range(8))
    test_set = []
    for i in idxs:
        print i
        handles_feas = handles_data[i]
        handles_pose = handles_poses[i]
        handles_tsr_split = handles_tsr_splits[i]
        plot_pose(handles_feas, handles_pose)
        handles_theta = get_theta_ground_truth(TRUE_MEAN, handles_feas, handles_tsr_split)
        test_set.append((np.copy(handles_theta), np.copy(handles_feas)))
    for i in idxs:
        print i + 8
        lock_feas = lock_data[i]
        lock_pose = lock_poses[i]
        lock_tsr_split = lock_tsr_splits[i]
        plot_pose(lock_feas, lock_pose)
        lock_theta = get_theta_ground_truth(TRUE_MEAN, lock_feas, lock_tsr_split)
        test_set.append((np.copy(lock_theta), np.copy(lock_feas)))
    np.savez("../data/user_study/subject%s_test_set.npz" % args.subject, test_set=test_set)
#########################################################
handles_datasets = []
lock_datasets = []
all_handles_data, handles_poses = np.array(preprocess_feasible(np.load('../data/handlebars_sim_data.npz')['data'], \
np.load('../data/handlebars_sim_data.npz')['poses'], False))
all_lock_data, lock_poses = np.array(preprocess_feasible(np.load('../data/bike_lock_sim_data.npz')['data'], \
np.load('../data/bike_lock_sim_data.npz')['poses'], False))

env, human, robot, target, target_desc = load_environment_file('../data/handlebars_problem_def.npz')
env.SetViewer('qtcoin')
newrobots = []
for ind in range(15):
    newrobot = RaveCreateRobot(env,human.GetXMLId())
    newrobot.Clone(human,0)
    for link in newrobot.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(0.8)
    newrobots.append(newrobot)
for link in robot.GetLinks():
    for geom in link.GetGeometries():
        geom.SetTransparency(0.8)
# collect_test_set(all_handles_data, handles_poses, all_lock_data, lock_poses)
# 1/0
def info_gain(dist, x):
    return (x, dist.info_gain(x, num_boxes=20))
def min_cost(dist, x):
    return (x, 0, dist.expected_cost2(x, dist.distribution_mode()))
if __name__ == '__main__':
    pool = mp.Pool(8)
    for set_idx in args.sets:
        np.random.seed(set_idx)
        handles_idxs = np.random.choice(len(all_handles_data), size=4)
        lock_idxs = np.random.choice(len(all_lock_data), size=4)
        handles_idxs[3] = 35
        print handles_idxs
        print lock_idxs
        handles_data = np.array(all_handles_data)[handles_idxs]
        lock_data = np.array(all_lock_data)[lock_idxs]
        data = [h for h in handles_data]
        for l in lock_data:
            data.append(l)
        data = np.array(data)
        test_set = np.copy(data)
        handles_chosen_poses = handles_poses[handles_idxs]
        lock_chosen_poses = lock_poses[lock_idxs]
        chosen_poses = [h for h in handles_chosen_poses]
        for l in lock_chosen_poses:
            chosen_poses.append(l)
        # collect_test_set(all_handles_data, handles_poses, all_lock_data, lock_poses)
        # collect_training_set(data, chosen_poses)
        # 1/0

        particles = []
        weights = []
        all_feas = []
        for feasible in data:
            all_feas.extend(feasible)
        all_feas = np.array(all_feas)
        mins = np.amin(all_feas, axis=0)
        maxes = np.amax(all_feas, axis=0)
        ranges = maxes - mins
        particles = np.random.uniform(0, 1, size=(NUM_PARTICLES, DOF))
        particles *= ranges
        particles += mins
        weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES
        weights = np.array(weights) / np.sum(weights)

        dist_active = SetWeightsParticleDistribution(np.copy(particles), np.copy(weights), cost, w=TRUE_WEIGHTS,\
        ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=RESAMPLING_VARIANCE)
        dist_passive = SetWeightsParticleDistribution(np.copy(particles), np.copy(weights), cost, w=TRUE_WEIGHTS,\
        ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O, h=RESAMPLING_VARIANCE)

        expected_costs, particles_passive = train_min_cost(dist_passive, data, chosen_poses)
        expected_infos, particles_active = train_active(dist_active, data, chosen_poses)

        pickle_dict = {'training_data': data, \
                        'distribution_active': dist_active, 'distribution_passive': dist_passive, \
                        'expected_infos': expected_infos,\
                        'expected_costs': expected_costs, 'training_poses': chosen_poses[:], \
                        'particles_active': particles_active, 'particles_passive': particles_passive}
        output = open('%s/subject%s_beleifs.pkl' % (DISTRIBUTION_DATA_FOLDER, args.subject), 'wb')
        pickle.dump(pickle_dict, output)
        output.close()

        collect_test_set(all_handles_data, handles_poses, all_lock_data, lock_poses)
        collect_training_set(data, chosen_poses)