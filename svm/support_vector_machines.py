import numpy as np
import matplotlib.pyplot as plt

# Creating two clouds of data from two different 2D gaussians
a_mean = np.array([-1, -2])
a_cov = np.eye(N=2, M=2)
A = np.random.multivariate_normal(mean=a_mean, cov=a_cov, size=(20, 2))

b_mean = np.array([0, 3])
B = np.random.multivariate_normal(mean=b_mean, cov=a_cov, size=(20, 2))

# TODO: Use SVM to find hyperplane separating the data

# Plotting the data
# TODO: Plot the hyperplane
plt.figure()
plt.scatter(x=A[:, 0], y=A[:, 1], label="A")
plt.scatter(x=B[:, 0], y=A[:, 1], label="B")
plt.legend()
plt.show()
