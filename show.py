import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def show_images_from_dataset(ds: tf.data.Dataset, n: int, **kwargs):
    """
    Show images from a tf.data.Dataset

    :param ds: tf.data.Dataset containing images
    :param n: number of images to show
    :param kwargs: parameters passed to matplotlib.pyplot.subplots
    :return:
    """
    x = [batch[0].numpy() for batch in ds.unbatch().batch(n * n).take(1)][0]
    x = np.clip(x, 0.0, 1.0)
    fig, ax = plt.subplots(n, n, **kwargs)
    for i in range(n):
        for j in range(n):
            ax[i, j].imshow(x[i + n * j])
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    pass


def show_similar_images(imgs, d2, num_images=10, num_pos=2, num_neg=1, figsize=(8, 20)):
    """
    Show similar and dissimilar images based on a distance matrix

    :param imgs: vector of images
    :param d2: distance matrix
    :param num_images: number of reference images to choose
    :param num_pos: number of similar images to show for each reference
    :param num_neg: number of similar images to show for each reference
    :param figsize: size of the figure
    :return:
    """
    fig, ax = plt.subplots(num_images, 1 + num_pos + num_neg, figsize=figsize)

    idxs = np.random.choice(len(imgs), size=num_images)

    ax[0, 0].set_title('Reference')
    for i in range(num_pos):
        ax[0, 1 + i].set_title(f'Pos {i}')
    for i in range(num_neg):
        ax[0, 1 + num_pos + i].set_title(f'Neg {i}')

    for k, i in enumerate(idxs):
        sort_idx = np.argsort(d2[i])
        sort_idx = sort_idx[:-1]

        ax[k, 0].imshow(imgs[i])
        ax[k, 0].get_xaxis().set_visible(False)
        ax[k, 0].get_yaxis().set_visible(False)

        for i in range(num_pos):
            ax[k, 1 + i].imshow(imgs[sort_idx[i]])
            ax[k, 1 + i].get_xaxis().set_visible(False)
            ax[k, 1 + i].get_yaxis().set_visible(False)

        for i in range(num_neg):
            ax[k, 1 + num_pos + i].imshow(imgs[sort_idx[-(1 + i)]])
            ax[k, 1 + num_pos + i].get_xaxis().set_visible(False)
            ax[k, 1 + num_pos + i].get_yaxis().set_visible(False)
