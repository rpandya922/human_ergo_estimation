import numpy as np
import sklearn.neighbors as neighbors
from random import shuffle
from scipy import stats
import matplotlib.pyplot as plt

def preprocess(data):
    X = np.array([np.array((s[0][0], s[0][2])) for s in data])
    y = np.array([np.array(s[1][:2]) for s in data])
    return X, y
def plot_xz(X, testX):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0], X[:,1], c='r', marker='o', s=20, label='train')
    ax.scatter(testX[:,0], testX[:,1], c='b', marker='s', s=20, label='test')
    plt.legend(loc='upper left')
    plt.show()
def plot_predictions(X, y, testX, testy, knn)):
    for i in range(len(testX)):
        x, y_query = testX[i], testy[i]
        ind = knn.kneighbors(x)[1][0]
        n = y[ind]
        pred = knn.predict([x])[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter((pred[0]), (pred[1]), c='k', marker='x', s=25, label='prediction')
        ax.scatter((y_query[0]), (y_query[1]), c='g', marker='^', s=20, label='query')
        ax.scatter(y[:,0], y[:,1], c='r', marker='o', s=20, label='train')
        ax.scatter(n[:,0], n[:,1], c='b', marker='s', s=25, label='neighbors')
        plt.legend(loc='upper left')
        plt.xlabel('shoulder joint angle')
        plt.ylabel('elbow joint angle')
        plt.show()
# train = np.load("./arm_joints_train.npy")
# test = np.load("./arm_joints_test.npy")
data = np.load("./arm_joints_bag_data.npy")[:107]
size = int(len(data) * 0.2)
shuffle(data)

train = data[size:]
test = data[:size]

X, y = preprocess(train)
testX, testy = preprocess(test)

knn = neighbors.KNeighborsRegressor(n_neighbors=3)
knn.fit(X, y)

print knn.score(testX, testy)
pred = knn.predict(testX)

diff = []
for i in range(len(testX)):
    p = pred[i]
    r = testy[i]
    print np.linalg.norm(r) ** 2
    diff.append(p - r)
    print np.linalg.norm(p - r) ** 2
    print "\n"