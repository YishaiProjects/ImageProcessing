import numpy as np
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray
from scipy.misc import imread as imread
import matplotlib.pyplot as plt
import os.path
from scipy.signal import convolve2d as conv2

GRAYSCALE = 1


def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation
    :param filename: the path of file
    :param representation: 1 represents grayscale 2 color image
    :return: the image file in desired representation
    """
    im = imread(filename)
    im_float = im.astype(np.float64)
    im_float = im_float / 255
    if representation == GRAYSCALE:
        return rgb2gray(im_float)
    else:
        return im_float


def fiter_gen(kernel_size):
    """
    gernerate a kernel acorrding to binomical coeeficients
    :param kernel_size:  the size of kernel to generate
    :return:  the new kernel matrix
    """
    one_d_ker = np.array([1])
    for i in range(0, kernel_size - 1):
        one_d_ker = np.convolve(one_d_ker, [1, 1])
    ker_sum = np.sum(one_d_ker)
    one_d_ker = one_d_ker / ker_sum
    return one_d_ker.reshape(1, kernel_size)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    creates a gaussian pyramid from image
    :param im: the image to create pyramid from
    :param max_levels: the levels of pyramid
    :param filter_size: the filter size to use
    :return: the pyramid and filter vector
    """
    filter_vec = fiter_gen(filter_size)
    pyr = []
    new_im = np.copy(im)
    pyr.append(new_im)
    for i in range(1, max_levels):
        if new_im.shape[0] <= 16 or new_im.shape[1] <= 16:
            break
        new_im = convolve(new_im, filter_vec, mode='reflect')
        new_im = convolve(new_im, filter_vec.T, mode='reflect')
        new_im = new_im[::2, 0::2]
        pyr.append(new_im)
    return pyr, filter_vec


def expand(im, filter_vec):
    """
    expand the image twice it size
    :param im: the image to expand
    :param filter_vec: the filter to use
    :param filter_size: the size of filter
    :return: new expanded image
    """
    expand_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    expand_im[::2, ::2] = im
    expand_im = convolve(expand_im, 2 * filter_vec, mode='reflect')
    return convolve(expand_im, 2 * filter_vec.T, mode='reflect')


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    creates a laplacian pyramid from image
    :param im: the image to create pyramid from
    :param max_levels: the levels of pyramid
    :param filter_size: the filter size to use
    :return: the pyramid and filter vector
    """
    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    laplace_pyr = []
    for i in range(gauss_pyr.__len__()):
        if i != gauss_pyr.__len__() - 1:
            laplace_pyr.append(gauss_pyr[i] - expand(gauss_pyr[i + 1], filter_vec))
        else:
            laplace_pyr.append(gauss_pyr[i])
    return laplace_pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    reconstruction of an image from its Laplacian Pyramid.
    :param lpyr: the pyramid
    :param filter_vec: the vector filter
    :param coeff: vector of coeeficitnt
    :return: the image reconstructed
    """
    image = expand(lpyr[lpyr.__len__() - 1] * coeff[lpyr.__len__() - 1], filter_vec)
    image += lpyr[lpyr.__len__() - 2]
    for i in range(lpyr.__len__() - 2, 0, -1):
        image = expand(image, filter_vec) + (lpyr[i - 1] * coeff[i - 1])
    return image


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    blends two images acording to given mask
    :param im1:th first image to blend
    :param im2: the image of the mask
    :param mask: the mask
    :param max_levels: how many levels the pyramid has
    :param filter_size_im: size of filter of blending
    :param filter_size_mask: the size of filter to use on mask
    :return: new blended image
    """
    L1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, filter_vec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask = mask.astype(np.float64)
    Gm, filter_vec = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    Lout = np.copy(Gm)
    for k in range(max_levels):
        Lout[k] = Gm[k] * L1[k] + (1 - Gm[k]) * L2[k]
    image = laplacian_to_image(Lout, filter_vec, np.array([1] * max_levels))
    image = np.clip(image, 0, 1)
    return image


def binary(img, bound):
    """
    transform an image to binary image given certain bound
    :param img:
    :param bound:
    :return:
    """
    return bound > img


def relpath(filename):
    """
    helper given fuction in ex
    :param filename:
    :return:
    """
    return os.path.join(os.path.dirname(__file__), filename)


def blur_spatial(im, kernel_size):
    """
    t performs image blurring using 2D convolution between the image f and a gaussian
    kernel
    :param im: the given image
    :param kernel_size: the size of kernel to be created in order to bluee image
    :return: the new blurred image
    """
    if kernel_size % 2 == 0:
        return -1
    else:
        one_d_ker = np.array([1])
        for i in range(0, kernel_size - 1):
            one_d_ker = np.convolve(one_d_ker, [1, 1])
        two_d_ker = np.outer(one_d_ker.T, one_d_ker)
        return conv2(im, two_d_ker, mode='same')