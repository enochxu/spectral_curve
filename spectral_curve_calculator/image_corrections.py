import numpy as np
from cv2 import cv2


# undo gamma correction
def gamma_expand(image):
    # conversion to float for increased accuracy
    u = image.astype("float32") / 255
    conds = [u <= 0.04045, u > 0.04045]
    funcs = [lambda u: u / 12.92, lambda u: np.power(((u + 0.055) / 1.055), 2.4)]
    linearized_image = np.piecewise(u, conds, funcs) * 255
    # conversion back to uint8 for opencv
    return linearized_image.astype("uint8")


# method creates vignette with a gaussian kernel(can be improved)
def de_vignette(image, sigma):
    # setting the gaussian deviation to match the size of the image
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, sigma)
    kernel_y = cv2.getGaussianKernel(rows, sigma)
    # sigma = gaussian standard deviation

    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()

    vignette = np.zeros_like(image)
    for i in range(3):
        vignette[:, :, i] = image[:, :, i] * (1 / mask)
    return vignette


def thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    retval, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def contour_mask(thresh):
    contour_mask = np.zeros(thresh.shape, np.uint8)
    kernel = np.ones((15, 15), np.uint8)

    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(contour_mask, contours, -1, 255, 2)
    return contour_mask
