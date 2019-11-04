import scipy.misc as misc
import numpy as np
import matplotlib.pyplot as plt
import skimage.color as color

img = misc.imread('my_image.jpg')

print(img)
print(img.dtype)
print(img.shape)
print(img.size)


#
# img = color.rgb2gray(img)
#
# print(img)
# print(img.dtype)
# print(img.shape)
# print(img.size)
#


def normalize(image):
    """
    A function that normalize the elements inside
    a numpy matrix, and change their type to float64.
    :param image: a numpy matrix
    :return: the matrix with all elements normalized
    to the range: [0, 1].
    """
    image = image.astype('float64')
    return image / 255  # here i divide by 255


def read_image(filename, representation):
    """
    A function that reads a file according to a
    given representation.
    :param filename: The file name.
    :param representation: 1 represents rgb,
    2 represents gray_scale.
    :return: An image according to the
    given representation.
    """
    image = misc.imread(filename)
    if representation == 2 or image.ndim != 3:  # here im checking
        # if im in rgb or gray with
        # the dimensions only
        if np.any(image > 1):
            return normalize(image)
    elif representation == 1:
        return color.rgb2gray(image)


def imdisplay(filename, representation):
    """
    A function that displays an image from a
    given file, according to the given representation.
    :param filename: The name of the given file.
    :param representation: The desired representation,
    1 for gray, 2 for rgb.
    :return: Nothing.
    """
    image = read_image(filename, representation)
    plt.imshow(image, cmap='gray')
    plt.show()


def mult_matrices(conversion_matrix, image):
    print("image: ", image)
    height = image.shape[0]
    width = image.shape[1]
    pixel_stack = np.transpose(np.reshape(image, (height * width, 3)))
    print("image after trans: ", image)
    result_pixel_stack = np.matmul(conversion_matrix, pixel_stack)
    print("image after mul: ", image)
    result_pixel_stack = np.transpose(result_pixel_stack)
    print("image after another trans: ", image)
    print("image at the end: ",
          np.reshape(result_pixel_stack, (height, width, 3)))
    return np.reshape(result_pixel_stack, (height, width, 3))


# I need to check that these functions are correct.
def rgb2yiq1(imRGB):
    """
    A function that converts an RGB matrix
    to YIQ matrix. (The RGB needs to be in the
    range: [0, 1] in order to se  the image good)
    :param imRGB: The given RGB matrix.
    :return: The YIQ matrix.
    """
    conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                  [0.596, -0.275, -0.321],
                                  [0.212, -0.523, 0.311]])
    return mult_matrices(conversion_matrix, imRGB)  #


# (This is another implementation of rgb2yiq)
def rgb2yiq2(imRGB):
    conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                  [0.596, -0.275, -0.321],
                                  [0.212, -0.523, 0.311]])
    height = imRGB.shape[0]
    width = imRGB.shape[1]
    pixel_stack = np.reshape(imRGB, (height * width, 3))
    l = []  # maybe fix here the list to
    # something else
    for i in range(3):
        Rs = pixel_stack[:, 0] * conversion_matrix[i, 0]
        Gs = pixel_stack[:, 0] * conversion_matrix[i, 1]
        Bs = pixel_stack[:, 0] * conversion_matrix[i, 2]
        l.append(np.add(np.add(Rs, Gs), Bs))
    result = np.vstack(l)
    result = np.transpose(result)
    return np.reshape(result, (height, width, 3))


def yiq2rgb(imYIQ):
    """
    A function that converts a YIQ image
    to RGB.
    :param imYIQ: the given image matrix.
    :return: The RGB representation of the image.
    """
    conversion_matrix = np.array([[1, 0.956, 0.619],
                                  [1, -0.272, -0.647],
                                  [1, -1.106, 1.703]])
    return mult_matrices(conversion_matrix, imYIQ)


img = yiq2rgb(rgb2yiq1(read_image('my_image.jpg', 2)))
plt.imshow(img, cmap='gray')
plt.show()
