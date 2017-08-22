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
#########################################################
# CONSTANTS AND FUNCTIONS
DOF = 4
NUM_PARTICLES = 1000
box_size = 0.5
ALPHA_I = 2.5
ALPHA_O = 2.5
TRUE_MEAN = np.array([0, 0, 0, 0])
TRUE_WEIGHTS = np.array([1, 1, 1, 1])
all_vars_dim1 = []
all_vars_dim2 = []
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
        target = env.ReadKinBodyXMLFile('../data/' + target_desc['object_file'])
        env.AddKinBody(target)
        target.SetTransform(object_start_pose)

    return env, human, robot, target, target_desc
def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)
def neg_log_likelihood(data, mean):
    total = 0
    for (theta, feasible) in data:
        total += pe.prob_theta_given_lam_stable_set_weight_num(theta, mean, TRUE_WEIGHTS, cost, 1)
        total -= pe.prob_theta_given_lam_stable_set_weight_denom(feasible, mean, TRUE_WEIGHTS, cost, 1)
    return -total
def plot_feas(data):
    fig, axes = plt.subplots(nrows=4, ncols=4)
    axes = np.ndarray.flatten(np.array(axes))
    fig2, axes2 = plt.subplots(nrows=4, ncols=4)
    axes2 = np.ndarray.flatten(np.array(axes2))
    for (i, (theta, feasible)) in enumerate(data):
        ax = axes[i]
        ax2 = axes2[i]

        ax.set_xlim(-3, 1.75)
        ax.set_ylim(-3, 1.5)
        ax2.set_xlim(-2, 1)
        ax2.set_ylim(-3, 0.5)

        ax.scatter(feasible[:,0], feasible[:,1], c='C0')
        ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
        ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)

        ax2.scatter(feasible[:,2], feasible[:,3], c='C0')
        ax2.scatter(theta[2], theta[3], c='C2', s=200, zorder=2)
        ax2.scatter(TRUE_MEAN[2], TRUE_MEAN[3], c='C3', s=200, zorder=2)
        plt.pause(0.2)
    fig.suptitle("Feasible sets to choose from (dim 1&2)")
    fig2.suptitle("Feasible sets to choose from (dim 3&4)")
    plt.pause(0.2)
def plot_belief(ax, particles, ground_truth, second=False):
    data_means = particles.T
    kernel = kde(data_means)
    if second:
        xx, yy = np.mgrid[-2:1:100j, -3:0.5:100j]
    else:
        xx, yy = np.mgrid[-3:1.75:100j, -3:1.5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='Greens')
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.scatter(ground_truth[0], ground_truth[1], c='C3', s=200, zorder=2)
def plot_belief_update(ax, particles, theta, feasible, ground_truth, second=False):
    plot_belief(ax, particles, ground_truth, second)
    ax.scatter(feasible[:,0], feasible[:,1], c='C0')
    ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
def plot_likelihood_heatmap(ax, theta, feasible, ground_truth, second=False, with_belief=False, dist=None):
    if second:
        xx, yy = np.mgrid[-2:1:100j, -3:0.5:100j]
    else:
        xx, yy = np.mgrid[-3:1.75:100j, -3:1.5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feasible)
    distances, indices = nbrs.kneighbors(feasible)
    max_dist = min(np.amax(distances), 0.5)
    distances, indices = nbrs.kneighbors(positions.T)
    def likelihood(idx, point):
        if distances[idx][0] >= max_dist:
            alpha = ALPHA_O
        else:
            alpha = ALPHA_I
        return pe.prob_theta_given_lam_stable_set_weight_num(theta, point, TRUE_WEIGHTS[:2], cost, alpha)\
        -pe.prob_theta_given_lam_stable_set_weight_denom(feasible, point, TRUE_WEIGHTS[:2], cost, alpha)
    likelihoods = np.array([likelihood(idx, p) for idx, p in enumerate(positions.T)])
    variance = np.var(likelihoods)
    f = np.reshape(likelihoods.T, xx.shape)
    if second:
        # print np.amin(likelihoods)
        # print "max: " + str(np.amax(likelihoods))
        ax.imshow(np.flip(f, 1).T, cmap='inferno', interpolation='nearest', extent=(-2, 1, -3, 0.5), vmin=-29.8, vmax=-0.9)
        all_vars_dim2.append(variance)
    else:
        # print np.amin(likelihoods)
        # print "max: " + str(np.amax(likelihoods))
        ax.imshow(np.flip(f, 1).T, cmap='inferno', interpolation='nearest', extent=(-3, 1.75, -3, 1.5), vmin=-54.6, vmax=-1.1)
        all_vars_dim1.append(variance)
        xx, yy = np.mgrid[-3:1.75:100j, -3:1.5:100j]
    ax.scatter(feasible[:,0], feasible[:,1], c='C0')
    ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
    if with_belief:
        if second:
            data_means = np.array(dist.particles)[:,2:].T
            kernel = kde(data_means)
            xx, yy = np.mgrid[-2:1:100j, -3:0.5:100j]
        else:
            data_means = np.array(dist.particles)[:,:2].T
            kernel = kde(data_means)
            xx, yy = np.mgrid[-3:1.75:100j, -3:1.5:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)
        cset = ax.contour(xx, yy, f, colors='k')
    ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)
    ax.set_title('variance: %0.2f, size: %d' % (variance, len(feasible)))
#########################################################
# f = np.load('../data/sim_rod_training_data.npz')
f = np.load('../data/sim_rod_weight_learning.npz')
# f = np.load('./sim_translation_training_data_varied.npz')
# idxs = np.random.choice(len(f['data']), size=8)
idxs = list(range(8))
print idxs
# idxs = [270, 281, 17, 3, 257, 160, 2]
# idxs = [0,1,2,3,4,5,6,7]
data_full = f['data_full'][idxs]
all_data = f['data'][idxs]
poses = f['poses'][idxs]
env, human, robot, target, target_desc = load_environment_file('../data/rod_full_problem_def.npz')
env.SetViewer('qtcoin')
data = all_data
fig, axes = plt.subplots(nrows=2, ncols=4)
axes = np.ndarray.flatten(np.array(axes))
fig2, axes2 = plt.subplots(nrows=2, ncols=4)
axes2 = np.ndarray.flatten(np.array(axes2))

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
for (i, (theta, feasible)) in enumerate(data):
# for (i, idx) in enumerate(idxs):
    # theta, feasible = data[idx]
    plot_likelihood_heatmap(axes[i], theta[:2], feasible[:,:2], TRUE_MEAN[:2])
    plot_likelihood_heatmap(axes2[i], theta[2:], feasible[:,2:], TRUE_MEAN[2:], second=True)
    target.SetTransform(poses[i])
    feas_full = data_full[i]
    with env:
        inds = np.array(np.linspace(0,len(feas_full)-1,15),int)
        for j,ind in enumerate(inds):
            newrobot = newrobots[j]
            env.Add(newrobot,True)
            newrobot.SetTransform(human.GetTransform())
            newrobot.SetDOFValues(feas_full[ind], human.GetActiveManipulator().GetArmIndices())
    env.UpdatePublishedBodies()
    plt.pause(0.01)
    # raw_input('Displaying pose ' + str(i) + ', press <Enter> to continue:')
fig.suptitle('dim 1&2 Particles: ' + str(NUM_PARTICLES) + ' alpha_i: ' + str(ALPHA_I) +\
             ' alpha_o: ' + str(ALPHA_O) + 'avg var: ' + str(np.average(all_vars_dim1)))
fig2.suptitle('dim 3&4 Particles: ' + str(NUM_PARTICLES) + ' alpha_i: ' + str(ALPHA_I) +\
             ' alpha_o: ' + str(ALPHA_O) + 'avg var: ' + str(np.average(all_vars_dim2)))
fig, axes = plt.subplots(nrows=1, ncols=2)
axes = np.ndarray.flatten(np.array(axes))
ax = axes[0]
ax.hist(all_vars_dim1, range=(0,140), normed=True)
ax = axes[1]
ax.hist(all_vars_dim2, range=(0, 40), normed=True)


for (i, (theta, feasible)) in enumerate(data):
    if len(feasible) > 1000:
        idxs = np.random.choice(len(feasible), size=1000)
        new_feas = feasible[idxs]
        data[i][1] = new_feas

particles = []
weights = []
all_feas = []
for (theta, feasible) in data:
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
dist = SetWeightsParticleDistribution(particles, weights, cost, w=TRUE_WEIGHTS, ALPHA_I=ALPHA_I, ALPHA_O=ALPHA_O)
# plot_feas(data)

def info_gain(dist, x):
    return (x, dist.info_gain(x[1], num_boxes=20), dist.expected_cost(x[1]))
if __name__ == '__main__':
    pool = mp.Pool(8)
    fig, axes = plt.subplots(nrows=4, ncols=3)
    axes = np.ndarray.flatten(np.array(axes))
    fig2, axes2 = plt.subplots(nrows=4, ncols=3)
    axes2 = np.ndarray.flatten(np.array(axes2))
    bar_fig, bar_axes = plt.subplots(nrows=4, ncols=3)
    bar_axes = np.ndarray.flatten(np.array(bar_axes))
    fig.suptitle('dim 1&2 Particles: ' + str(NUM_PARTICLES) + ' alpha_i: ' + str(ALPHA_I) +\
                 ' alpha_o: ' + str(ALPHA_O))
    fig2.suptitle('dim 3&4 Particles: ' + str(NUM_PARTICLES) + ' alpha_i: ' + str(ALPHA_I) +\
                 ' alpha_o: ' + str(ALPHA_O))
    ax = axes[0]
    ax2 = axes2[0]

    plot_belief(ax, np.array(dist.particles)[:,:2], TRUE_MEAN[:2])
    plot_belief(ax2, np.array(dist.particles)[:,2:], TRUE_MEAN[2:], second=True)
    plt.pause(0.2)

    for i in range(1, 12):
        func = partial(info_gain, dist)
        pooled = pool.map(func, data)
        print
        expected_infos = [sample[1] for sample in pooled]
        expected_costs = [sample[2] for sample in pooled]
        max_idx = np.argmax(expected_infos)
        # max_idx = np.argmin(expected_costs)
        (theta, feasible) = pooled[max_idx][0]
        actual_infos = []
        ent_before = dist.entropy(num_boxes=20)
        for j in range(len(data)):
            t, f = data[j]
            d = SetWeightsParticleDistribution(dist.particles, dist.weights, dist.cost, dist.w, dist.ALPHA_I, dist.ALPHA_O)
            d.weights = d.reweight(t, f)
            actual_infos.append(ent_before - d.entropy(num_boxes=20))
        # (theta, feasible) = data[i]
        dist.weights = dist.reweight_vectorized(theta, feasible)
        dist.resample()

        target.SetTransform(poses[max_idx])
        # target.SetTransform(poses[i])
        feas_full = data_full[max_idx]
        # feas_full = data_full[i]
        with env:
            inds = np.array(np.linspace(0,len(feas_full)-1,15),int)
            for j,ind in enumerate(inds):
                newrobot = newrobots[j]
                env.Add(newrobot,True)
                newrobot.SetTransform(human.GetTransform())
                newrobot.SetDOFValues(feas_full[ind], human.GetActiveManipulator().GetArmIndices())
        env.UpdatePublishedBodies()

        ax = axes[i]
        ax2 = axes2[i]
        bar_ax = bar_axes[i]
        # plot_belief_update(ax, np.array(dist.particles)[:,:2], theta, feasible, TRUE_MEAN)
        # plot_belief_update(ax2, np.array(dist.particles)[:,2:], theta[2:], feasible[:,2:], TRUE_MEAN[2:], second=True)
        plot_likelihood_heatmap(ax, theta[:2], feasible[:,:2], TRUE_MEAN[:2], second=False, with_belief=True, dist=dist)
        plot_likelihood_heatmap(ax2, theta[2:], feasible[:,2:], TRUE_MEAN[2:], second=True, with_belief=True, dist=dist)
        bar_ax.bar(np.arange(len(data)), expected_infos, 0.35, color='C0', label='expected info gain')
        bar_ax.bar(np.arange(len(data)) + 0.35, actual_infos, 0.35, color='C1', label='actual info gain')
        bar_ax.bar(max_idx, expected_infos[max_idx], 0.35, color='C2', label='chosen set expected info')
        plt.pause(0.2)
        # raw_input('Displaying iteration ' + str(i) + ', press <Enter> to continue:')
    plt.show()
