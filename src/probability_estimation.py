import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal as mvn

#########################################################
# CONSTANTS AND FUNCTIONS
ALPHA = 0.1
feasible = []
def normalize(angles):
    s = []
    for x in angles:
        while x < -3.14:
            x += 6.28
        while x > 3.14:
            x -= 6.28
        s.append(x)
    return np.array(s)
def preprocess(data):
    # X = np.array([np.array((s[0][0], s[0][2])) for s in data])
    # y = np.array([np.array(s[1][:2]) for s in data])
    X = np.array([np.array((s[0][0], s[0][2])) for s in data])
    y = np.array([np.array(s[1][:3]) for s in data])
    return X, y
def create_feasible_set(theta, stddev):
    f = [np.random.normal(4, stddev, theta.shape) + theta for _ in range(25)]
    global feasible
    feasible.extend(f)
    f.append(y)
    return np.array(f)
def distance_cost(theta, theta_star, w):
    """
    Computes the cost of set of joint angles based on squared distance from optimal

    @param theta: the joint angles to evaluate
    @param theta_star: the optimal joint angles (i.e. nominal arm configuration)
    @param w: learned weights for the joints
    @return: the cost for this configuration
    """
    l = np.diag(w)
    return np.dot(theta - theta_star, np.dot(l, theta - theta_star))
def prob_y_given_lam(theta, theta_star, w, Theta_x, cost):
    """
    Computes the probability of some joint angles given their weights

    @param theta: the joint angles to evaluate
    @param theta_star: the optimal/nominal joint configuration
    @param w: weights for the joints
    @param Theta_x: the set of all feasible joint angles for this object
    @param cost: a function which takes three paramters: y, y_star and lam to compute y's cost
    @return: the probability of y given lam
    """
    p = np.exp(-ALPHA * cost(theta, theta_star, w))
    denom = 0
    for theta_prime in Theta_x:
        r = np.exp(-ALPHA * cost(theta_prime, theta_star, w))
        denom += r
    if denom == 0:
        return 0.99
    return p / np.sum([np.exp(-ALPHA * cost(theta_prime, theta_star, w)) for theta_prime in Theta_x])
def prob_lam_given_y(theta, lam, Theta_x, cost, prior):
    """
    Computes a value proportional to the probability of lambda given theta

    @param y: the joint angles to evaluate
    @param y_star: the optimal/nominal joint configuration
    @param lam: weights for the joints concatonated with optimal joint angles
    @param cost: a function which takes three paramters: y, y_star and lam to compute y's cost
    @param prior: a function of the prior distribution over lambda
    @return: value proportional to the probability of lambda given y
    """
    w = np.array(lam[:3]) / np.linalg.norm(lam[:3])
    theta_star = lam[3:]
    return prob_y_given_lam(theta, theta_star, w, Theta_x, cost) * prior(lam)
#########################################################

data = np.load('./arm_joints_feasible_data.npy')[:1]
feasible_sets = np.load("./feasible_sets2.npy")
X, ys = preprocess(data)
avg = np.mean(ys, axis=0)
print avg
# y_star = [0.75719418, 0.30581145, 0.07879616]
# x_ax = np.arange(0, 4, 0.1)
# y1 = []
# y2 = []
# y3 = []
# y4 = []
# for v in x_ax:
#     y_star[1] = v
def cost(y, y_star, lam):
    return distance_cost(y, y_star, lam)
new_data = []
for i in range(len(X)):
    x, y = X[i], normalize(ys[i])
    Y_x = create_feasible_set(y, 1)
    # Y_x = feasible_sets[i]
    # Y_x = np.vstack((Y_x, y))
    new_data.append((x, y, Y_x))
var = mvn(mean=[0,0,0,0,0,0], cov=np.diag([1000, 1000, 1000, 20, 20, 20]))
def prior(vec):
    return var.pdf(vec)
def objective(lam):
    return -np.sum([np.log(prob_lam_given_y(theta, lam, Theta_x, cost, prior)) for (x, theta, Theta_x) in new_data])
res = minimize(objective, [0, 1, 0, 0, 0, 0])
l = res.x
print "weights: " + str(l[:3])
print "theta*: " + str(l[3:])
y_star = l[3:]

ys = []
for d in data:
    ys.append(d[1][:3])
ys = np.array(ys)
fs = np.array(feasible)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=ys[:,0], ys=ys[:,1], zs=ys[:,2], s=200, label='optimal')
ax.scatter(y_star[0], y_star[1], y_star[2], c='r', s=300, label='optimized')
ax.scatter(xs=fs[:,0], ys=fs[:,1], zs=fs[:,2], c='g', marker='^', s=100, label='feasible')
plt.legend(loc='upper left')
plt.show()

# fs = []
# for f in feasible_sets:
#     fs.extend(f)
# fs = np.array(fs)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xs=fs[:,0], ys=fs[:,1], zs=fs[:,2], s=20)
# ax.scatter(y_star[0], y_star[1], y_star[2], c='r', s=25)
# plt.show()

    # norm = np.linalg.norm(l)
    # y1.append(norm)
    # l = l / norm
    # y2.append(l[0])
    # y3.append(l[1])
    # y4.append(l[2])
# plt.plot(x_ax, y1)
# plt.xlabel("std dev")
# plt.ylabel("lambda norm")
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x_ax, y2, label='lambda[0]')
# ax.plot(x_ax, y3, label='lambda[1]')
# ax.plot(x_ax, y4, label='lambda[2]')
# plt.legend(loc='upper left')
# plt.xlabel('y_star[1]')
# plt.ylabel('lambda val')
# plt.show()