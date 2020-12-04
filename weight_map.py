import numpy as np

from skimage.transform import resize

from skimage.filters.rank import entropy
from skimage.morphology import square

from pyramids import convolve

def normalized_local_entropy(image, window_size):
    """
    A fonction that computes the local entropy given an image and a window size

    :param image: The grayscale image
    :param window_size: The size of the window that determines the neighbourhood of a pixel, an integer
    :return: The local entropy of the image, a grayscale image
    """

    local_entropy = 0.125*entropy(image, square(window_size))
    return local_entropy

def local_contrast(image, window_size):
    """
     A fonction that computes the local contrast given an image and a window size

    :param image: The grayscale image
    :param window_size: The size of the window that determines the neighbourhood of a pixel, an integer
    :return: The local contrast of the image, a grayscale image
    """

    conv_filter = np.ones((window_size,window_size), dtype=int)
    local_mean = convolve(image, conv_filter)/(window_size**2)
    contrast = np.zeros((image.shape[0], image.shape[1]))
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            patch = image[max(0, x-int(window_size/2)):min(image.shape[0], x+int(window_size/2)), max(0, y-int(window_size/2)):min(image.shape[1], y+int(window_size/2))]
            patch = np.square(patch - local_mean[x,y])
            contrast[x,y] = np.sqrt(np.sum(patch)/(window_size**2))
    return contrast

