import numpy as np
from cv2 import cv2

#undo gamma correction
def gamma_expand(image):
    #conversion to float for increased accuracy
    u = image.astype('float32') / 255
    conds = [u <= 0.04045, u > 0.04045]
    funcs = [lambda u: u / 12.92, lambda u: np.power(((u + 0.055) / 1.055), 2.4)]
    linearized_image = np.piecewise(u, conds, funcs) * 255
    #conversion back to uint8 for opencv
    return linearized_image.astype('uint8')


