from __future__ import division
import sys
sys.path.insert(0, '../')
from openravepy import *
import prpy
import numpy as np
import seaborn
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
import probability_estimation as pe
from distribution import SetMeanParticleDistribution
import multiprocessing as mp
from functools import partial
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal as mvn
import readline

l1 = 3
l2 = 3
ALPHA = 10
ALPHA_I = 1
ALPHA_O = 1
TRUE_MEAN = np.array([0, 0])
TRUE_WEIGHTS = np.array([1, 3])
TRUE_WEIGHTS = TRUE_WEIGHTS / np.linalg.norm(TRUE_WEIGHTS)
two_pi = 2 * np.pi
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
def prefilled_input(prompt, prefill=''):
   readline.set_startup_hook(lambda: readline.insert_text(prefill))
   try:
      return raw_input(prompt)
   finally:
      readline.set_startup_hook()
def cost(theta, theta_star, w):
    d_theta = np.square(theta - theta_star)
    return d_theta.dot(w)
def create_line(pt1, pt2, step_size=0.1):
    points = []
    m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    if pt1[0] < pt2[0]:
        x = pt1[:]
        while x[0] <= pt2[0]:
            points.append(x[:])
            x[0] += step_size
            x[1] += m * step_size
    else:
        x = pt2[:]
        while x[0] <= pt1[0]:
            points.append(x[:])
            x[0] += step_size
            x[1] += m * step_size
    print len(points)
    return np.array(points)
def create_box(upper_left, lower_right, box_size=0.5):
    feas = []
    x = upper_left[:]
    while x[1] >= lower_right[1]:
        while x[0] <= lower_right[0]:
            feas.append(x[:])
            x[0] += box_size
        x[0] = upper_left[0]
        x[1] -= box_size
    return np.unique(feas, axis=0)
def create_ellipse(x0, y0, a, b):
    rand = np.random.uniform(0, 1, size=(1000, 2)) * np.array([2*a, 2*b])
    rand += np.array([x0 - a, y0 - b])
    feas = []
    vals = np.sum(((rand - np.array([x0, y0])) / np.array([a, b])) ** 2, axis=1)
    for i, v in enumerate(vals):
        if v <= 1:
            feas.append(rand[i])
    return np.array(feas)
def get_theta(x, y):
    inv_cos = np.arccos( ((l1**2) + (l2**2) - (x**2) - (y**2)) / (2 * l1 * l2))
    theta_prime = np.arcsin(l2 * np.sin(inv_cos) / np.sqrt((x**2) + (y**2)))
    theta1_partial = np.arctan2(x, y)
    theta2 = np.pi - inv_cos
    if np.isnan(inv_cos) or np.isnan(theta_prime):
        return []
    # thetas = [[theta1_partial + theta_prime, -theta2], [theta1_partial - theta_prime, theta2]]
    thetas = [[theta1_partial - theta_prime, theta2]]
    if thetas[0][0] > np.pi:
        thetas[0][0] -= two_pi
    elif thetas[0][0] < -np.pi:
        thetas[0][0] += two_pi
    # if thetas[1][0] > np.pi:
    #     thetas[1][0] -= two_pi
    # elif thetas[1][0] < -np.pi:
    #     thetas[1][0] += two_pi
    return thetas
def create_sample(feas):
    nums = pe.prob_theta_given_lam_stable_set_weight_num(feas, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA)
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feas, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA)
    probs = np.exp(nums - denom)
    chosen_idx = np.argmax(probs)
    chosen = feas[chosen_idx]
    return (chosen, feas)
def create_sample2(feas, weights):
    nums = pe.prob_theta_given_lam_stable_set_weight_num(feas, TRUE_MEAN, weights, cost, ALPHA)
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feas, TRUE_MEAN, weights, cost, ALPHA)
    probs = np.exp(nums - denom)
    chosen_idx = np.argmax(probs)
    chosen = feas[chosen_idx]
    return (chosen, feas)
def plot_objects(data):
    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.suptitle("'objects' in xy space")
    axes = np.ndarray.flatten(np.array(axes))
    for (i, feasible) in enumerate(data):
        ax = axes[i]
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.scatter(feasible[:,0], feasible[:,1])
    plt.pause(0.2)
def create_sample_from_xy(obj):
    feas = []
    for (x, y) in obj:
        feas.extend(get_theta(x, y))
    feas = np.array(feas)
    nums = pe.prob_theta_given_lam_stable_set_weight_num(feas, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA)
    denom = pe.prob_theta_given_lam_stable_set_weight_denom(feas, TRUE_MEAN, TRUE_WEIGHTS, cost, ALPHA)
    probs = np.exp(nums - denom)
    chosen_idx = np.argmax(probs)
    chosen = feas[chosen_idx]
    return (chosen, feas)
def plot_feas(data, weights=None):
    if weights is None:
        weights = [TRUE_WEIGHTS] * len(data)
    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.suptitle("feasible sets in theta space, l1: 3, l2: 3")
    axes = np.ndarray.flatten(np.array(axes))
    for (i, (theta, feasible)) in enumerate(data):
        w = weights[i]
        ax = axes[i]
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.scatter(feasible[:,0], feasible[:,1])
        ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
        ax.scatter(TRUE_MEAN[0], TRUE_MEAN[1], c='C3', s=200, zorder=2)
        ax.set_title('weights: ' + str(w))
    plt.pause(0.2)
def plot_belief(ax, particles, ground_truth, alpha=0.01):
    # data_weights = particles.T
    # kernel = kde(data_weights)
    # points = sample_spherical(700).T
    # new_pts = []
    # for p in points:
    #     if p[0] >= 0 and p[1] >= 0:
    #         new_pts.append(p)
    # points = np.array(new_pts)
    # colors = kernel(points.T)
    # ax.scatter(points[:,0], points[:,1], c=colors, cmap='viridis')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.scatter(np.array(particles)[:,0], np.array(particles)[:,1], c='g', alpha=alpha)
    ax.scatter(ground_truth[0], ground_truth[1], c='C3', s=200, zorder=2)
def plot_belief_update(ax, particles, theta, feasible, ground_truth):
    plot_belief(ax, particles, ground_truth)
    ax.scatter(feasible[:,0], feasible[:,1], c='C0')
    ax.scatter(theta[0], theta[1], c='C2', s=200, zorder=2)
def get_min_max_likelihood(data):
    mins = []
    maxes = []
    for (theta, feasible) in data:
        xx, yy = np.mgrid[0:1.25:100j, 0:1.25:100j]
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
            return pe.prob_theta_given_lam_stable_set_weight_num(theta, TRUE_MEAN, point, cost, alpha)\
            -pe.prob_theta_given_lam_stable_set_weight_denom(feasible, TRUE_MEAN, point, cost, alpha)
        likelihoods = np.array([likelihood(idx, p) for idx, p in enumerate(positions.T)])
        mins.append(np.amin(likelihoods))
        maxes.append(np.amax(likelihoods))
    return min(mins), max(maxes)
def plot_likelihood_heatmap(ax, theta, feasible, ground_truth, vmin=-27, vmax=-2, with_belief=False, dist=None):
    xx, yy = np.mgrid[0:1.25:100j, 0:1.25:100j]
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
        return pe.prob_theta_given_lam_stable_set_weight_num(theta, TRUE_MEAN, point, cost, alpha)\
        -pe.prob_theta_given_lam_stable_set_weight_denom(feasible, TRUE_MEAN, point, cost, alpha)
    likelihoods = np.array([likelihood(idx, p) for idx, p in enumerate(positions.T)])
    f = np.reshape(likelihoods.T, xx.shape)
    ax.imshow(np.flip(f, 1).T, cmap='inferno', interpolation='nearest', extent=(0, 1.25, 0, 1.25), vmin=vmin, vmax=vmax)
    if with_belief:
        data_means = np.array(dist.particles).T
        kernel = kde(data_means)
        xx, yy = np.mgrid[0:1.25:100j, 0:1.25:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)
        cset = ax.contour(xx, yy, f, colors='k')
    ax.scatter(ground_truth[0], ground_truth[1], c='C3', s=200, zorder=2)
def sample_spherical(npoints, ndim=2):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec
def plot_likelihood_heatmap_norm_weights(ax, theta, feasible, ground_truth, vmin=-5, vmax=0):
    points = sample_spherical(700).T
    new_pts = []
    for p in points:
        if p[0] >= 0 and p[1] >= 0:
            new_pts.append(p)
    points = np.array(new_pts)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(feasible)
    distances, indices = nbrs.kneighbors(feasible)
    max_dist = min(np.amax(distances), 0.5)
    distances, indices = nbrs.kneighbors(points)
    def likelihood(idx, point):
        if distances[idx][0] >= max_dist:
            alpha = ALPHA_O
        else:
            alpha = ALPHA_I
        return pe.prob_theta_given_lam_stable_set_weight_num(theta, TRUE_MEAN, point, cost, alpha)\
        -pe.prob_theta_given_lam_stable_set_weight_denom(feasible, TRUE_MEAN, point, cost, alpha)
    likelihoods = np.array([likelihood(idx, p) for idx, p in enumerate(points)])
    print min(likelihoods)
    print 'max: ' + str(max(likelihoods))
    colors = np.random.random(100)
    ax.scatter(points[:,0], points[:,1], c=likelihoods, vmin=vmin, vmax=vmax, cmap='inferno')
    ax.scatter(ground_truth[0], ground_truth[1], c='C3', s=200, zorder=2)
def prob_of_truth(dist, ground_truth):
    DOF = len(dist.particles[0])
    cov = np.diag(np.ones(DOF)) * 0.0625
    likelihoods = mvn.pdf(dist.particles, mean=ground_truth, cov=cov)
    return np.sum(likelihoods) / dist.NUM_PARTICLES
def dist_to_truth(dist, ground_truth):
    # mean = np.mean(dist.particles, axis=0)
    mode = dist.distribution_mode()
    return np.linalg.norm(mode - ground_truth)
def plot_metric(data, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data)
    ax.set_xlabel('iteration')
    ax.set_ylabel(label)