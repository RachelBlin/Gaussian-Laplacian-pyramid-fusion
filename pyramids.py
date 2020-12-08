import numpy as np

from scipy.signal import convolve2d

def gaussian_pyramid(image, kernel, levels):
    """
    A function to create a Gaussian pyramid of a defined number of levels and from a chosen kernel.

    :param image: The image we want to use of dimension (N,M,3) or (M,N)
    :param kernel: The Gaussian kernel of dimention (k,k)
    :param levels: The desired number of levels in the Gaussian pyramid, an integer
    :return: The Gaussian pyramid, a list of numpy arrays
    """

    if len(np.shape(image)) == 3:
        gauss_l_r = image[:, :, 0]
        gauss_l_g = image[:, :, 1]
        gauss_l_b = image[:, :, 2]
    gauss_l = image
    pyramid = [gauss_l]
    for l in range(levels):
        if len(np.shape(image)) == 3:
            # channels last format
            gauss_l_r = downsample(gauss_l_r, kernel)
            gauss_l_g = downsample(gauss_l_g, kernel)
            gauss_l_b = downsample(gauss_l_b, kernel)
            gauss_l = np.zeros((gauss_l_b.shape[0], gauss_l_b.shape[1], 3))
            gauss_l[:, :, 0] = gauss_l_r
            gauss_l[:, :, 1] = gauss_l_g
            gauss_l[:, :, 2] = gauss_l_b
        else:
            gauss_l = downsample(gauss_l, kernel)
        pyramid.append(gauss_l)
    return pyramid

def laplacian_pyramid(image, kernel, levels):
    """
    A function to create a Laplacian pyramid of a defined number of levels and from a chosen kernel.

    :param image: The image we want to use of dimension (N,M,3) or (M,N)
    :param kernel: The Gaussian kernel of dimention (k,k)
    :param levels: The desired number of levels in the Laplacian pyramid, an integer
    :return: The Laplacian pyramid, a list of numpy arrays
    """

    gauss = gaussian_pyramid(image, kernel, levels)
    pyramid = []
    for l in range(len(gauss) - 2, -1, -1):
        if len(np.shape(image)) == 3:
            # channels last format
            gauss_l1r = upsample(gauss[l+1][:, :, 0])
            gauss_l1g = upsample(gauss[l+1][:, :, 1])
            gauss_l1b = upsample(gauss[l+1][:, :, 2])
            if gauss_l1r.shape[0] > gauss[l][:, :, 0].shape[0]:
                gauss_l1r = np.delete(gauss_l1r, -1, axis=0)
                gauss_l1g = np.delete(gauss_l1g, -1, axis=0)
                gauss_l1b = np.delete(gauss_l1b, -1, axis=0)
            if gauss_l1r.shape[1] > gauss[l][:, :, 0].shape[1]:
                gauss_l1r = np.delete(gauss_l1r, -1, axis=1)
                gauss_l1g = np.delete(gauss_l1g, -1, axis=1)
                gauss_l1b = np.delete(gauss_l1b, -1, axis=1)
            lap_l_r = gauss[l][:, :, 0] - gauss_l1r
            lap_l_g = gauss[l][:, :, 1] - gauss_l1g
            lap_l_b = gauss[l][:, :, 2] - gauss_l1b
            lap_l = np.zeros((lap_l_r.shape[0], lap_l_r.shape[1], 3))
            lap_l[:, :, 0] = lap_l_r
            lap_l[:, :, 1] = lap_l_g
            lap_l[:, :, 2] = lap_l_b
        else:
            gauss_l1 = upsample(gauss[l+1])
            if gauss_l1.shape[0] > gauss[l].shape[0]:
                gauss_l1 = np.delete(gauss_l1, -1, axis=0)
            if gauss_l1.shape[1] > gauss[l].shape[1]:
                gauss_l1 = np.delete(gauss_l1, -1, axis=1)
            lap_l = gauss[l] - gauss_l1
        pyramid.append(lap_l)
    return pyramid

def fused_laplacian_pyramid(gauss_pyramid_mod1, gauss_pyramid_mod2, lap_pyramid_mod1, lap_pyramid_mod2):
    """
    A funtion that builds a fused Laplacian pyramid of two modalities of the same image

    :param gauss_pyramid_mod1: The Gaussian pyramid of modality 1, a list of grayscale images, the first one in highest resolution
    :param gauss_pyramid_mod2: The Gaussian pyramid of modality 2, a list of grayscale images, the first one in highest resolution
    :param lap_pyramid_mod1: The Laplacian pyramid of modality 1, a list of grayscale images, the last one in highest resolution
    :param lap_pyramid_mod2: The Laplacian pyramid of modality 2, a list of grayscale images, the last one in highest resolution
    :return: The fused Laplacian pyramid of two modalities, a list of grayscale images, the last one in highest resolution,
    """

    fused_laplacian = []
    len_lap = len(lap_pyramid_mod1)
    for l in range(len_lap):
        fused_laplacian_temp = gauss_pyramid_mod1[len_lap-l-1]*lap_pyramid_mod1[l] + gauss_pyramid_mod2[len_lap-l-1]*lap_pyramid_mod2[l]
        fused_laplacian.append(fused_laplacian_temp)
    return fused_laplacian

def collapse_pyramid(lap_pyramid, gauss_pyramid):
    """
    A function to collapse a Laplacian pyramid in order to recover the enhanced image

    :param lap_pyramid: A Laplacian pyramid, a list of grayscale images, the last one in highest resolution
    :param gauss_pyramid: A Gaussian pyramid, a list of grayscale images, the last one in lowest resolution
    :return: A grayscale image
    """

    image = lap_pyramid[0]
    if len(np.shape(image)) == 3:
        im_r = upsample(gauss_pyramid[-1][:, :, 0])
        im_g = upsample(gauss_pyramid[-1][:, :, 1])
        im_b = upsample(gauss_pyramid[-1][:, :, 2])
        if im_r.shape[0] > image.shape[0]:
            im_r = np.delete(im_r, -1, axis=0)
            im_g = np.delete(im_g, -1, axis=0)
            im_b = np.delete(im_b, -1, axis=0)
        if im_r.shape[1] > image.shape[1]:
            im_r = np.delete(im_r, -1, axis=1)
            im_g = np.delete(im_g, -1, axis=1)
            im_b = np.delete(im_b, -1, axis=1)
        gauss = np.zeros((im_r.shape[0], im_r.shape[1], 3))
        gauss[:, :, 0] = im_r
        gauss[:, :, 1] = im_g
        gauss[:, :, 2] = im_b
    else:
        gauss = upsample(gauss_pyramid[-1])
        if gauss.shape[0] > image.shape[0]:
            gauss = np.delete(gauss, -1, axis=0)
        if gauss.shape[1] > image.shape[1]:
            gauss = np.delete(gauss, -1, axis=1)
    image = image + gauss
    for l in range(1,len(lap_pyramid),1):
        if len(np.shape(image)) == 3:
            im_r = upsample(image[:, :, 0])
            im_g = upsample(image[:, :, 1])
            im_b = upsample(image[:, :, 2])
            if im_r.shape[0] > lap_pyramid[l].shape[0]:
                im_r = np.delete(im_r, -1, axis=0)
                im_g = np.delete(im_g, -1, axis=0)
                im_b = np.delete(im_b, -1, axis=0)
            if im_r.shape[1] > lap_pyramid[l].shape[1]:
                im_r = np.delete(im_r, -1, axis=1)
                im_g = np.delete(im_g, -1, axis=1)
                im_b = np.delete(im_b, -1, axis=1)
            pyr_upsampled = np.zeros((im_r.shape[0], im_r.shape[1], 3))
            pyr_upsampled[:, :, 0] = im_r
            pyr_upsampled[:, :, 1] = im_g
            pyr_upsampled[:, :, 2] = im_b
        else:
            pyr_upsampled = upsample(image)
            if pyr_upsampled.shape[0] > lap_pyramid[l].shape[0]:
                pyr_upsampled = np.delete(pyr_upsampled, -1, axis=0)
            if pyr_upsampled.shape[1] > lap_pyramid[l].shape[1]:
                pyr_upsampled = np.delete(pyr_upsampled, -1, axis=1)
        image = lap_pyramid[l] + pyr_upsampled
    return image

def convolve(image, kernel):
    """
    A fonction to perform a 2D convolution operation over an image using a chosen kernel.

    :param image: The grayscale image we want to use of dimension (N,M)
    :param kernel: The convolution kernel of dimention (k,k)
    :return: The convolved image of dimension (N,M)
    """
    im_out = convolve2d(image, kernel, mode='same', boundary='symm')
    return im_out

def downsample(image, kernel):
    """
    A function to downsample an image.

    :param image: The grayscale image we want to use of dimension (N,M)
    :param kernel: The Gaussian blurring kernel of dimention (k,k)
    :return: The downsampled image of dimension (N/factor,M/factor)
    """
    blur_image = convolve(image, kernel)
    img_downsampled = blur_image[::2, ::2]
    return img_downsampled

def upsample(image):
    """

    :param image: The grayscale image we want to use of dimension (N,M)
    :param factor: The upsampling factor, an integer
    :return: The upsampled image of dimension (N*factor,M*factor)
    """

    #kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/12
    kernel = smooth_gaussian_kernel(0.4)

    img_upsampled = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
    img_upsampled[::2, ::2] = image[:, :]
    img_upsampled = 4 * convolve(img_upsampled, kernel)
    return img_upsampled

def classical_gaussian_kernel(k, sigma):
    """
    A function to generate a classical Gaussian kernel

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
