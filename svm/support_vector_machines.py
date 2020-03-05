import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -------------------- Data generation --------------------
# Creating two clouds of data from two different 2D gaussians
a_mean = np.array([-1, -2])
a_cov = np.eye(N=2, M=2)*2
A = np.random.multivariate_normal(mean=a_mean, cov=a_cov, size=40)

b_mean = np.array([0, 3])
B = np.random.multivariate_normal(mean=b_mean, cov=a_cov, size=40)

all_data = np.concatenate((A, B), axis=0)

# -------------------- SVM with hinge loss --------------------

# Initial parameters
w0 = np.array([0, 1]).reshape(-1, 1)
b0 = np.array([1])
p0 = np.array([w0[0], w0[1], b0])


# Affine function for hyperplane
def aff(x, w, b):
    return w.T@x - b


def y(data_point):
    if data_point in A:
        return 1
    elif data_point in B:
        return -1
    else:
        print("Unknown data.")


# Loss function
def empirical_hinge_loss(parameters):
    w = parameters[:2]
    b = parameters[2]
    N = len(all_data)
    loss = np.sum([max([0, 1 - y(x) * aff(x, w, b)]) for x in all_data])/N
    return loss


# Minimize loss to observe optimal hyperplane (affine function) parameters
res = minimize(fun=empirical_hinge_loss, x0=p0).x
w_star = res[:2]
b_star = res[2]


# -------------------- Plots --------------------
# Plotting the data
plt.figure()
plt.scatter(x=A[:, 0], y=A[:, 1], label="A")
plt.scatter(x=B[:, 0], y=B[:, 1], label="B")
# TODO: Plot SVM decision boundary
# x = np.arange(-5, 5, 0.1).reshape(-1, 1)
# plt.contour(x, x, np.array([aff(z, w_star, b_star) for z in np.concatenate((x, x), axis=0)]), levels=[0])
plt.legend()
plt.show()
