import math
import numpy as np
import cv2 as cv2

def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 316138411

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """

    return np.convolve(in_signal, k_size, mode='full')


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    # Define kernels for computing derivatives in x and y directions
    kx = np.array([-1, 0, 1])
    ky = np.array([-1, 0, 1]).reshape((3, 1))

    # Compute derivatives in x and y directions using conv2D function
    Ix = conv2D(in_image, kx)
    Iy = conv2D(in_image, ky)

    # Compute magnitude and direction matrices
    magnitude = np.sqrt(np.square(Ix) + np.square(Iy))
    direction = np.arctan2(Iy, Ix)

    return direction, magnitude


def get_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Compute a 2D Gaussian kernel with the given kernel size and standard deviation
    :param kernel_size: Size of the kernel (should be odd)
    :param sigma: Standard deviation of the Gaussian distribution
    :return: A 2D Gaussian kernel
    """
    k = (kernel_size - 1) // 2
    x = np.arange(-k, k + 1)
    kernel = np.exp(-np.square(x) / (2 * np.square(sigma))) / (sigma * np.sqrt(2 * np.pi))
    kernel = kernel / np.sum(kernel)
    kernel_2d = np.outer(kernel, kernel)
    return kernel_2d

def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    sigma = k_size / 6
    kernel = get_gaussian_kernel(k_size, sigma)
    blurred_image = conv2D(in_image, kernel)
    return blurred_image


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    sigma = k_size / 6
    kernel = cv2.getGaussianKernel(k_size, sigma)
    blurred_image = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return blurred_image


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
     ############ i dont implement this function ##############
    return img



def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    # Convert the input image to grayscale if it is in color
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    lap_img = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
    laplacian_8bit = cv2.convertScaleAbs(lap_img)

    # the zero-crossings
    _, thresholded = cv2.threshold(laplacian_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    zero_cross = cv2.ximgproc.thinning(thresholded, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    return zero_cross

def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
      Find Circles in an image using a Hough Transform algorithm extension
      To find Edges you can Use OpenCV function: cv2.Canny
      :param img: Input image
      :param min_radius: Minimum circle radius
      :param max_radius: Maximum circle radius
      :return: A list containing the detected circles,
                  [(x,y,radius),(x,y,radius),...]
    """

    img = cv2.GaussianBlur(img, (5, 5), 1)
    # Apply Canny edge detection
    img = cv2.Canny(img.astype(np.uint8), 255 / 3, 255)
    circles = []
    height, width = img.shape
    # Find edge points
    edges = np.argwhere(img == 255)
    accumulator = np.zeros((height, width, max_radius - min_radius + 1))

    # Find the points
    min_radius = min_radius if min_radius > 0 else 1  # Adjust min_radius if it's less than or equal to 0
    theta_values = np.arange(0, 360, 5)  # Generate theta values
    # Compute the corresponding x, y, and r values using vectorized operations
    x = (min_radius * np.cos(theta_values * np.pi / 180)).astype(int)
    y = (min_radius * np.sin(theta_values * np.pi / 180)).astype(int)
    r = np.arange(min_radius, max_radius + 1)
    # Create the Points array using broadcasting
    Points = np.column_stack((np.repeat(x, len(r)), np.repeat(y, len(r)), np.tile(r, len(theta_values))))


    # for r in range(min_radius, max_radius + 1):
    #     for t in range(0, 360, 5):
    #         x = int(r * np.cos(t * np.pi / 180))
    #         y = int(r * np.sin(t * np.pi / 180))
    #         Points.append((x, y, r))

    for i, j in edges:
        for x, y, r in Points:
            b = j - y
            a = i - x
            if 0 <= a < height and 0 <= b < width:
                accumulator[a, b, r - min_radius] += 1

    (h, w, rad) = accumulator.shape
    # Calculate the threshold value
    # t_value =np.median([np.max(accumulator[:, :, radius]) for radius in range(rad)])
    t_value = np.median(accumulator) * 0.8
    # Iterate over the accumulator matrix
    for r in range(rad):
        for i in range(h):
            for j in range(w):
                if accumulator[i, j, r] >= t_value:
                    circles.append((j, i, r + min_radius))
    return circles

def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    # Generate a 2D Gaussian kernel for spatial filtering
    kernel = get_gaussian_kernel(k_size, sigma_space)
    # Initialize output image
    out_image = np.zeros_like(in_image)
    # Pad the input image
    padded_image = np.pad(in_image, ((k_size // 2, k_size // 2), (k_size // 2, k_size // 2)), mode='constant')
    # Loop over every pixel in the input image
    for i in range(in_image.shape[0]):
        for j in range(in_image.shape[1]):
            # Get the current pixel value
            center = padded_image[i:i + k_size, j:j + k_size]
            # Compute the Gaussian weights for color filtering
            color_weights = np.exp(-np.square(center - in_image[i, j]) / (2 * np.square(sigma_color)))
            # Compute the bilateral filter weights
            weights = kernel * color_weights
            # Normalize the weights
            weights /= np.sum(weights)
            # Compute the filtered output value
            out_image[i, j] = np.sum(weights * center)
    # Compute the OpenCV implementation
    cv_image = cv2.bilateralFilter(in_image.astype(np.float32), k_size, sigma_color, sigma_space)
    return cv_image, out_image
