import numpy as np
import sklearn.neighbors as neighbors
from random import shuffle

def preprocess(data):
    X = np.array([np.array(s[0]) for s in data])
    y = np.array([np.array(s[1][:2]) for s in data])
    return X, y

train = np.load("./arm_joints_train.npy")
test = np.load("./arm_joints_test.npy")

shuffle(train)
size = int(len(train) * 0.2)
train = train[size:]

X, y = preprocess(train)
testX, testy = preprocess(test)

knn = neighbors.KNeighborsRegressor(n_neighbors=3)
knn.fit(X, y)
print knn.score(testX, testy)
