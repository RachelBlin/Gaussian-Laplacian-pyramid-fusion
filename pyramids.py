import numpy as np

from scipy.signal import convolve2d

from skimage.transform import resize

def gaussian_pyramid(image, kernel, levels, factor):
    """
    A function to create a Gaussian pyramid of a defined number of levels and from a chosen kernel.

    :param image: The image we want to use of dimension (N,M,3) or (M,N)
    :param kernel: The Gaussian kernel of dimention (k,k)
    :param levels: The desired number of levels in the Gaussian pyramid, an integer
    :param factor: The downsampling factor, an integer
    :return: The Gaussian pyramid, a list of numpy arrays
    """

    gauss_l = image
    pyramid = [gauss_l]
    for l in range(levels):
        if len(np.shape(image)) == 3:
            # channels last format
            gauss_l[:, :, 0] = downsample(convolve(gauss_l[:, :, 0], kernel), factor)
            gauss_l[:, :, 1] = downsample(convolve(gauss_l[:, :, 1], kernel), factor)
            gauss_l[:, :, 2] = downsample(convolve(gauss_l[:, :, 2], kernel), factor)
        else:
            gauss_l = downsample(convolve(gauss_l, kernel), factor)
        pyramid.append(gauss_l)
    return pyramid

def laplacian_pyramid(image, kernel, levels, factor):
    """
    A function to create a Laplacian pyramid of a defined number of levels and from a chosen kernel.

    :param image: The image we want to use of dimension (N,M,3) or (M,N)
    :param kernel: The Laplacian kernel of dimention (k,k)
    :param levels: The desired number of levels in the Laplacian pyramid, an integer
    :param factor: The upsampling factor, an integer
    :return: The Laplacian pyramid, a list of numpy arrays
    """

    gauss = gaussian_pyramid(image, kernel, levels, factor)
    pyramid = []
    for l in range(len(gauss) - 2, -1, -1):
        if len(np.shape(image)) == 3:
            # channels last format
            corrected_shape = upsample(gauss[l+1][:, :, 0], factor).shape
            lap_l[:, :, 0] = resize(gauss[l][:, :, 0], (corrected_shape[0], corrected_shape[1])) - 4*convolve(upsample(gauss[l+1][:, :, 0], factor), kernel)
            lap_l[:, :, 1] = resize(gauss[l][:, :, 1], (corrected_shape[0], corrected_shape[1])) - 4*convolve(upsample(gauss[l+1][:, :, 1], factor), kernel)
            lap_l[:, :, 2] = resize(gauss[l][:, :, 2], (corrected_shape[0], corrected_shape[1])) - 4*convolve(upsample(gauss[l+1][:, :, 2], factor), kernel)
        else:
            corrected_shape = upsample(gauss[l+1], factor).shape
            lap_l = resize(gauss[l], (corrected_shape[0], corrected_shape[1])) - convolve(upsample(gauss[l+1], factor), kernel)
        pyramid.append(lap_l)
    return pyramid

def convolve(image, kernel):
    """
    A fonction to perform a 2D convolution operation over an image using a chosen kernel.

    :param image: The grayscale image we want to use of dimension (N,M)
    :param kernel: The convolution kernel of dimention (k,k)
    :return: The convolved image of dimension (N,M)
    """
    im_out = convolve2d(image, kernel, mode='same', boundary='symm')
    return im_out

def downsample(image, factor):
    """
    A function to downsample an image.

    :param image: The grayscale image we want to use of dimension (N,M)
    :param factor: The downsampling factor, an integer
    :return: The downsampled image of dimension (N/factor,M/factor)
    """
    img_downsampled = image[::factor, ::factor]
    return img_downsampled

def upsample(image, factor):
    """

    :param image: The grayscale image we want to use of dimension (N,M)
    :param factor: The upsampling factor, an integer
    :return: The upsampled image of dimension (N*factor,M*factor)
    """

    img_upsampled = np.zeros((image.shape[0]*factor, image.shape[1]*factor), dtype=np.float64)
    img_upsampled[::factor, ::factor] = image[:, :]
    return img_upsampled

def classical_gaussian_kernel(k, sigma):
    """
    A function to generate a Gaussian kernel for smooth filtering

    :param k: The size of the kernel, an integer
    :param sigma: variance of the gaussian distribution
    :return: A Gaussian kernel, a numpy array of shape (k,k)
    """
    w = np.linspace(-(k - 1) / 2, (k - 1) / 2, k)
    x, y = np.meshgrid(w, w)
    kernel = 0.5*np.exp(-0.5*(x**2 + y**2)/(sigma**2))/(np.pi*sigma**2)
    return kernel

def smooth_gaussian_kernel(a):
    """
     A 5*5 gaussian kernel to perform smooth filtering.

    :param a: the coefficient of the smooth filter. A float usually within [0.3, 0.6]
    :return: A smoothing Gaussian kernel, a numpy array of shape (5,5)
    """
    w = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
    kernel = np.outer(w, w)
    return kernel
