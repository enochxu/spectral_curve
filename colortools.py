import numpy as np

#undo gamma correction
def gamma_expand(image):
    u = image / 255
    conds = [u <= 0.04045, u > 0.04045]
    funcs = [lambda u: u / 12.92, lambda u: np.power(((u + 0.055) / 1.055), 2.4)]
    linearized_image = np.piecewise(u, conds, funcs) * 255
    return linearized_image

