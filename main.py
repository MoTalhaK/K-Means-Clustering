import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rc('axes', titlesize=30)

plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.figsize'] = (16, 9)

data = pd.read_csv("twodpoints.txt", sep=',', names=["a", "b"])
# print(data.shape)
data.head()


#
# f1 = data['a'].values
# f2 = data['b'].values
# X = np.array(list(zip(f1, f2)))
# plt.scatter(f1, f2, c="black", s=200)
# plt.show()

# QUESTION 3 A)
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def find_closest(D, center):
    clu = np.zeros(D.shape[0])
    for i in range(D.shape[0]):
        d = np.square(np.sum(abs(D[i] - center) ** 2, axis=1))
        clu[i] = np.argmin(d)
    return clu


def find_mean(D, clust, k):
    d, n = D.shape
    centr = np.zeros((k, n))
    for i in range(k):
        x = D[clust == i]
        if x.shape[0] > 0:
            average = np.mean(x, axis=0)
            centr[i, :] = average
    return centr


def k_means(D, k, init, type, iters, style):
    n = D.shape[0]
    d = D.shape[1]
    labels = np.zeros((n, 1))
    cMat = np.zeros(n)
    # type of initialization
    if type == "manual":
        C = init
        # initial centroids
        plt.scatter(C[:, 0], C[:, 1], marker='X', s=400, linewidths=3, c='g')
    elif type == "uniform":
        C = np.random.uniform(np.min(D), np.max(D), size=(k, d))
    elif type == "max":
        # first centre chose uniformly at random
        C = np.random.uniform(np.min(D), np.max(D), size=(k, d))

        # each next center is chosen to be the data point that maximizes
        # the sum of the (euclidean) distances from the previous data points.
        if k > 1:
            for i in range(2, k):
                maxSum = -math.inf
                maxPoint = np.zeros(n)

                for j in range(1, np.size(D, 0)):
                    sum = 0
                    point = D[j, :]
                    for m in range(1, i - 1):
                        centerM = cMat[m]
                        distX = np.linalg.norm(centerM - point)
                        sum = sum + distX

                    if sum > maxSum:
                        maxSum = sum
                        maxPoint = point
                        C[i] = maxPoint
    print(C)

    C_old = np.zeros((iters, C.shape[0], C.shape[1]))

    clusters = np.zeros(n)
    for t in range(iters):
        C_old[t, :] = C
        clusters = find_closest(D, C)
        C = find_mean(D, clusters, k)
        # assign labels from 1 to k to each point in a cluster
        for i in range(1, n):
            distance = np.zeros(k)
            for j in range(1, k):
                distance[j] = np.linalg.norm(D[i, :] - C[j, :])

            minDistance = math.inf
            for label in range(1, k):
                if distance[label] <= minDistance:
                    minDistance = distance[label]
                    labels[i, 0] = label

        for a in range(1, k):
            total = 0
            c = 1
            for b in range(1, n):
                if a == labels[b, :]:
                    total = total + D[b, :]
                    c += 1
            C[a, :] = total / c

    color = cm.rainbow(np.linspace(0, 1, k))
    if style == "2D":
        if type == "manual":
            plt.title(str(k) + " means manual (Method 1)")
            file_string = "manual"
        elif type == "uniform":
            plt.title(str(k) + " means uniform (Method 2)")
            file_string = "uniform"
        elif type == "max":
            plt.title(str(k) + f" means max distance (Method 3)")
            file_string = "max"
        plt.scatter(D[:, 0], D[:, 1], c=color[clusters.astype(int), :], s=200)
        if C.shape[1] > 1:
            plt.scatter(C[:, 0], C[:, 1], c='#050505', marker='*', s=600)
            # save plots to directory "output"
            plt.savefig("output/" + str(k) + f"_means_" + file_string + ".png")
            # plt.show()
    elif style == "3D":
        fig = plt.figure()
        ax = Axes3D(fig)
        if type == "manual":
            plt.title(str(k) + " means manual (Method 1)")
            file_string = "manual"
        elif type == "uniform":
            plt.title(str(k) + " means uniform (Method 2)")
            file_string = "uniform"
        elif type == "max":
            plt.title(str(k) + f" means max distance (Method 3)")
            file_string = "max"
        ax.scatter(D[:, 0], D[:, 1], D[:, 2], c=color[clusters.astype(int), :], s=200)
        plt.savefig("output/" + str(k) + f"_means_" + file_string + ".png")
        # plt.show()

    return C, clusters, labels


def cost_function(D, clustLabel, centers):
    error = 0
    for i in range(math.floor(clustLabel.max()) + 1):
        points = np.where(clustLabel == i)
        center = centers[i]
        for j in range(len(points[0])):
            error += np.linalg.norm((D[points[0][j]] - center)) ** 2

    return error


# QUESTION 3 B)
# initial = np.array([[-5, 12.5], [10, 5], [-2.5 , 0]]) # 3 means
# C, clusters = k_means(3, initial, "manual", 100)

# initial = np.array([[-2, 4], [5, 11], [-3 , 2]]) # 3 means
# C, clusters = k_means(3, initial, "manual", 100)

# initial = np.array([[-2, 4], [5, 11], [-3, 2], [-2.5, 12.5]])  # 4 means
# C, clusters = k_means(4, initial, "manual", 100)

# initial = np.array([[2.5, 10], [5, 11], [-3, 2], [-2.5, 12.5]])
# C, clusters = k_means(4, initial, "manual", 100)

costs = np.zeros(11)

# QUESTION 3 D)

X = np.loadtxt("twodpoints.txt", delimiter=',')

for i in range(1, 11):
    C, clusters, labels = k_means(X, i, 0, "max", 50, "2D")
    cost = cost_function(X, labels, C)
    costs[i] = cost

plt.xlabel("K", fontsize=30)
plt.ylabel("Cost", fontsize=30)
plt.plot(costs, 'bx-')
plt.savefig("output/k_means_cost.png")
plt.show()

# QUESTION 3 E)

# X = np.loadtxt("threedpoints.txt", delimiter=',')
#
# for i in range(1, 11):
#     C, clusters, labels = k_means(X, i, 0, "max", 50, "3D")
#     cost = cost_function(X, labels, C)
#     costs[i] = cost