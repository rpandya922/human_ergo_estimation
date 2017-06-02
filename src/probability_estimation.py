import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#########################################################
# CONSTANTS AND FUNCTIONS
ALPHA = 0.5
def preprocess(data):
    X = np.array([np.array((s[0][0], s[0][2])) for s in data])
    y = np.array([np.array(s[1][:2]) for s in data])
    return X, y
def create_feasible_set(y, stddev):
    f = [np.random.normal(0, stddev, y.shape) + y for _ in range(25)]
    f.append(y)
    return np.array(f)
def distance_cost(y, y_star, lam):
    """
    Computes the cost of set of joint angles based on squared distance from optimal

    @param y: the joint angles to evaluate
    @param y_star: the optimal joint angles (i.e. nominal arm configuration)
    @param lam: learned weights for the joints
    @return: the cost for this configuration
    """
    l = np.diag(lam)
    return np.dot(y - y_star, np.dot(l, y - y_star))
def prob_y_given_lam(y, lam, y_x, cost):
    """
    Computes the probability of some joint angles given their weights

    @param y: the joint angles to evaluate
    @param lam: weights for the joints
    @param y_x: the set of all feasible joint angles for this object
    @param cost: a function which takes two paramters: y and lam to compute y's cost
    @return: the probability of y given lam
    """
    p = np.exp(-ALPHA * cost(y, lam))
    return p / np.sum([np.exp(-ALPHA * cost(y_prime, lam)) for y_prime in y_x])
def prob_lam_given_y(y, lam, y_x, cost, prior):
    """
    Computes a value proportional to the probability of lambda given y

    @param y: the joint angles to evaluate
    @param lam: weights for the joints
    @param cost: a function which takes two paramters: y and lam to compute y's cost
    @param prior: a function of the prior distribution over lambda
    @return: value proportional to the probability of lambda given y
    """
    return prob_y_given_lam(y, lam, y_x, cost) * prior(lam)
#########################################################

data = np.load('./arm_joints_bag_data.npy')[:107]
X, ys = preprocess(data)
y_star = np.mean(ys, axis=0)
x_ax = np.arange(0, 4, 0.1)
y1 = []
y2 = []
y3 = []
for v in x_ax:
    y_star[1] = v
    def cost(y, lam):
        return distance_cost(y, y_star, lam)
    new_data = []
    for i in range(len(X)):
        x, y = X[i], ys[i]
        Y_x = create_feasible_set(y, 1)
        new_data.append((x, y, Y_x))
    def objective(lam):
        return -np.sum([np.log(prob_lam_given_y(y, lam, Y_x, cost, lambda x: 1)) for (x, y, Y_x) in new_data])
    res = minimize(objective, [0, 0])
    l = res.x
    norm = np.linalg.norm(l)
    y1.append(norm)
    l = l / norm
    y2.append(l[0])
    y3.append(l[1])
# plt.plot(x_ax, y1)
# plt.xlabel("std dev")
# plt.ylabel("lambda norm")
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_ax, y2, label='lambda[0]')
ax.plot(x_ax, y3, label='lambda[1]')
plt.legend(loc='upper left')
plt.xlabel('y_star[1]')
plt.ylabel('lambda val')
plt.show()