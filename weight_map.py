import numpy as np

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

    local_entropy = entropy(image, square(window_size))
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

def visibility(image, kernel1, kernel2):
    """
    A fonction that computes the visibility of an image given an image and two gaussian kernel

    :param image: The grayscale image
    :param kernel1: The gaussian kernel to compute the blurred image
    :param kernel2: The gaussian kernel to perform the final step of the visibility
    :return: The visibility, a grayscale image
    """

    img_blur = convolve(image, kernel1)
    visibility = np.sqrt(convolve(np.square(image - img_blur), kernel2))
    return visibility

def weight_combination(entropy, contrast, visibility, alpha1, alpha2, alpha3):
    """
    Combining the entropy, the contrast and the visibility to build a weight layer

    :param entropy: The local entropy of the image, a grayscale image
    :param contrast: The local contrast of the image, a grayscale image
    :param visibility: The visibility of the image, a grayscale image
    :param alpha1: The weight of the local entropy, a float within [0, 1]
    :param alpha2: The weight of the local contrast, a float within [0, 1]
    :param alpha3: The weight of the visibility, a float within [0, 1]
    :return: Weight map of the image, a grayscale image
    """

    weight = entropy**alpha1 * contrast**alpha2 * visibility**alpha3
    return weight

def weight_normalization(weight1, weight2):
    """
    A function to normalize the weights of each modality so the weights' sum is 1 for each pixel of the image

    :param weght1: The weight of madality 1, a grayscale image
    :param weight2: The weight of modality 2, a grayscale image
    :return: Two weights, weight1_normalized and weight2_normalized, respectively the normalized versions of weight1 and weight2, two grayscale images.
    """

    weight1_normalized = weight1 / (weight1 + weight2)
    weight2_normalized = weight2 / (weight1 + weight2)
    return weight1_normalized, weight2_normalized



