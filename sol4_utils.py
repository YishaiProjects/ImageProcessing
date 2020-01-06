from scipy.signal import convolve2d
import numpy as np
import imageio
import skimage.color as color
import scipy.ndimage.filters


def normalize(image):
    """
    A function that normalize the elements inside
    a numpy matrix, and change their type to float64.
    :param image: a numpy matrix
    :return: the matrix with all elements normalized
    to the range: [0, 1].
    """
    image = image.astype('float64')
    return image / 255


def read_image(filename, representation):
    """
    A function that reads a file according
    to a given representation.
    :param filename: The file name.
    :param representation: 1 represents rgb,
    2 represents gray_scale.
    :return: An image according to the
    given representation.
    """
    image = imageio.imread(filename)
    if representation == 2 or image.ndim != 3:
        if np.any(image > 1):
            return normalize(image)
    elif representation == 1:
        return color.rgb2gray(image)


def normalize_gaussian_filter(filter_vec):
    """
    A function that normalize the gaussian
    filter.
    :param filter_vec: The filter.
    :return: the normalized filter.
    """
    return filter_vec / np.sum(filter_vec)


def build_gaussian_filter(filter_size):
    """
    A function that build a gaussian filter
    according to a given size.
    :param filter_size: odd integer that
    represents the desired size.
    :return: The gaussian filter as
    a numpy array with shape: [1, filter_size].
    (without normalization)
    """
    number_of = filter_size - 2
    if filter_size > 1:
        small_kernel = np.array([1, 1])
    else:
        small_kernel = np.array([1])
    the_filter = small_kernel
    for i in range(number_of):
        the_filter = np.convolve(the_filter, small_kernel)
    return np.reshape(the_filter, (1, filter_size))


def convolve(im, filter_vec):
    """
    A function that convolve an image with
    a given filter.
    :param im: The given image as a grayscale
    image with double values in [0, 1].
    :param filter_vec: The filter as a row vector.
    :return: The image after the deletion.
    """
    im_after_rows_convolution = scipy.ndimage.filters.convolve(
        im, filter_vec)
    return scipy.ndimage.filters.convolve(
        im_after_rows_convolution, np.transpose(filter_vec))


def delete_elements(im):
    """
    A function that delete all the elements
    in the odd indexes. (as well as all the
    odd rows)
    :param im: The given image.
    :return: The image after the deletion.
    """
    odd_indexes_of_rows = np.arange(1, im.shape[0] + 1, 2)
    odd_indexes_of_cols = np.arange(1, im.shape[1] + 1, 2)
    im_after_horizontal_clean = np.delete(im,
                                          odd_indexes_of_rows, axis=0)
    im_after_vertical_clean = np.delete(im_after_horizontal_clean,
                                        odd_indexes_of_cols,
                                        axis=1)
    return im_after_vertical_clean


def reduce(im, filter_vec):
    """
    A function that creates a smaller image
    (smaller by factor of 2) of the given image.
    :param im: The given image.
    :param filter_vec: The given filter
    that blur the image.
    :return: The smaller image.
    """
    im_after_convolution = convolve(im, filter_vec)
    return delete_elements(im_after_convolution)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    A function that construct a Gaussian
    pyramid of a given image.
    :param im: A grayscale image with double
    values in [0, 1].
    :param max_levels: The maximal number of levels1
    in the resulting pyramid.
    :param filter_size: The size of the Gaussian filter
    (an odd scalar that represents a squared filter) to be used
    in constructing the pyramid filter.
    :return: The resulting pyramid pyr as a standard python array,
    and filter_vec - which is a row vector of shape (1, filter_size).
    """
    filter_vec = build_gaussian_filter(filter_size)
    filter_vec = normalize_gaussian_filter(filter_vec)
    pyr = [im]
    smallest_image = im
    max_levels -= 1
    while max_levels >= 1 and smallest_image.shape[0] > 16 \
            and smallest_image.shape[1] > 16:
        smallest_image = reduce(smallest_image, filter_vec)
        pyr.append(smallest_image)
        max_levels -= 1
    return pyr, filter_vec


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


# another pyramid functions:                                 maybe delete this at the end
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


def build_gaussian_pyramid_n(im, max_levels, filter_size):
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
        new_im = convolve2d(new_im, filter_vec)          # i deleted from this line and the next: mode='reflect'
        new_im = convolve2d(new_im, filter_vec.T)
        new_im = new_im[::2, 0::2]
        pyr.append(new_im)
    return pyr, filter_vec
