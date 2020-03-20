import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -------------------- Data generation --------------------
# Creating two clouds of data from two different 2D gaussians
a_mean = np.random.uniform(low=-4.0, high=4.0, size=(2,))
a_cov = np.eye(N=2, M=2)*2
A = np.random.multivariate_normal(mean=a_mean, cov=a_cov, size=40)

b_mean = np.random.uniform(low=-4.0, high=4.0, size=(2,))
B = np.random.multivariate_normal(mean=b_mean, cov=a_cov, size=40)

all_data = np.concatenate((A, B), axis=0)

# -------------------- SVM with hinge loss --------------------

# Initial parameters
w0 = np.array([0, 1]).reshape(-1, 1)
b0 = np.array([1])
p0 = np.array([w0[0], w0[1], b0])


def y(data_point):
    if data_point in A:
        return 1
    elif data_point in B:
        return -1
    else:
        print("Unknown data.")


# Affine function for hyperplane
def aff(x, w, b):
    return w.T@x - b


# Loss function
# TODO: Add regularisation term
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
plt.figure(figsize=(12, 6))
plt.scatter(x=A[:, 0], y=A[:, 1], label="A")
plt.scatter(x=B[:, 0], y=B[:, 1], label="B")
# Plot SVM decision boundary
xs = np.linspace(-5, 5)
a = - w_star[0] / w_star[1]  # ??
ys = a*xs - b_star/w_star[1]
plt.plot(xs, ys, color='black', label='decision boundary')
plt.legend()
plt.show()
