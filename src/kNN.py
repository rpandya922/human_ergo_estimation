import numpy as np
import sklearn.neighbors as neighbors
from random import shuffle

def preprocess(data):
    X = np.array([np.array(s[0]) for s in data])
    y = np.array([s[1][3:] for s in data])
    return X, y
def predict(pred, real):
    for i in range(len(pred)):
        p = pred[i]
        y = real[i]
        print min([np.linalg.norm(p - y), np.linalg.norm(-p - y)])

train = np.load("./hand_robot_training_data.npy")
test = np.load("./hand_robot_test_data.npy")

shuffle(train)
size = int(len(train) * 0.2)
val = train[:size]
# train = train[size:]

X, y = preprocess(train)
valX, valy = preprocess(val)
testX, testy = preprocess(test)

knn = neighbors.KNeighborsRegressor(n_neighbors=1)
knn.fit(X, y)

# predict(knn.predict(valX), valy)
# predict(knn.predict(testX), testy)

print np.var(testy, axis=0)
pred = knn.predict(testX)
diff = []
for i in range(len(pred)):
    p = pred[i]
    r = testy[i]
    diff.append(min([p - r, -p - r], key=np.linalg.norm))
print np.var(diff, axis=0)
