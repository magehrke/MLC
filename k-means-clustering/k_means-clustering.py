# K-means clustering
# Maximilian Alexander Gehrke

import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt

np.set_printoptions(edgeitems=100)


# Attention:    When subtracting two rgb values, the output is modulo 255
#               this is because the image has uint values when importing.

#               Our workaround is that we transform the pixels to np.int
#               when calculating the norms.

def k_means(img_arr, k, delta, max_iter, verbose=False):
    """
    THis method performs the k-means algorithm on an image and outputs the
    image with the colours of the closest mean.

    :param img_arr: numpy array; the original image
    :param k: int; the number of means & number of colours o the output image
    :param delta: float; the convergence criterion. If the norm of the old and
        the current mu's is below delta, the algorithm stops.
    :param max_iter: int; the algorithm stops if we have max_iter iterations.
    :param verbose: if True, prints are output to the console
    :return: numpy array; the output image
    """

    # Save the original shape of the image
    img_shape = img_arr.shape
    # Calculate the flattened shape of the image
    flat_shape = (img_shape[0] * img_shape[1], img_shape[2])
    # Flatten the image
    img_res = np.reshape(img_arr, flat_shape)
    # Get the number of pixels
    pixel_num = img_res.shape[0]

    # ----------------------------------------------------------------------- #

    mu_old = None

    # Initialize the means
    rand_indices = np.random.choice(pixel_num, k)
    mu = np.asarray([img_res[x, :] for x in rand_indices])
    print("--- Initialization  ---\n{}".format(mu))
    sys.stdout.flush()

    labels = np.empty(pixel_num, dtype=np.int)

    for u in range(max_iter):
        print("\n--- Update {} ---".format(u))
        sys.stdout.flush()

        # --- EXPECTATION --- #
        for i in tqdm(range(pixel_num)):

            pix = img_res[i]

            closest_l2 = calc_norm(mu[0], pix)
            closest_label = 0

            for kk in range(1, k):
                l2 = calc_norm(mu[kk], pix)

                if l2 < closest_l2:
                    closest_l2 = l2
                    closest_label = kk

            labels[i] = closest_label

        # --- MAXIMIZATION --- #
        mu_old = mu.copy()
        for kk in range(k):
            kk_bools = np.where(labels == kk)[0]
            kk_rgb = img_res[kk_bools]

            mu[kk] = np.sum(kk_rgb, axis=0) / len(kk_rgb)

            print("Mu {}: {}".format(kk, mu[kk]))
            sys.stdout.flush()

        # ------------------------------------------------------------------- #
        # Color each pixel in the colour off the nearest mixture component (mu)

        img_save = img_res.copy()

        for i, label in enumerate(labels):
            img_save[i] = mu[label]

        # ------------------------------------------------------------------- #

        # Reshape the image to its original shape
        img_save = np.reshape(img_save, img_shape)
        plt.imsave("data/town_out_iter_" + str(u) + ".jpg", img_save)
        sys.stdout.flush()

        # ------------------------------------------------------------------- #
        # Check convergence

        convergence = 0
        for i in range(k):
            convergence += calc_norm(mu_old[i], mu[i])

        print("Convergence: {}".format(convergence))
        sys.stdout.flush()

        if convergence < delta:
            break

    return img_arr


def calc_norm(vec_1, vec_2):
    vec_1_np = np.asarray(vec_1, dtype=np.int)
    vec_2_np = np.asarray(vec_2, dtype=np.int)
    assert vec_1_np.shape == vec_2_np.shape

    return np.linalg.norm(vec_1_np - vec_2_np)


# --------------------------------------------------------------- #

# Import the picture as a numpy array
im = plt.imread("data/town.jpg").copy()

# Run the k-means algorithm on the image
im = k_means(img_arr=im, k=2, delta=10, max_iter=20, verbose=False)



