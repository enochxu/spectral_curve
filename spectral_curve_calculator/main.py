import numpy as np
import image_corrections as ic
import cv2
from tkinter import Tk, filedialog

sigma = 900
thresh_val = 40


class Sample:
    def __init__(self, name):
        self.name = name
        self.image = open_image()


def change_sigma(new_sigma):
    global sigma
    sigma = new_sigma


def change_thresh(new_thresh):
    global thresh_val
    thresh_val = new_thresh


def resize(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# let user choose file, returns image
def open_image():
    root = Tk()
    root.fileName = filedialog.askopenfilename(
        initialdir="./samples",
        title="Select an Image",
        filetypes=((".jpg", "*.jpg"), (".png", "*.png"), ("all files", "*.*")),
    )
    fname = root.fileName
    image = cv2.imread(fname)
    return image


def crop_image(image):
    r = cv2.selectROI(image)
    cropped = image[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
    return cropped


def main():
    image = open_image()
    linearized_image = ic.gamma_expand(image) * 2
    # resize image
    image = resize(image, 0.3)
    linearized_image = resize(linearized_image, 0.3)
    # learn to auto-adjust brightness

    global sigma
    cv2.namedWindow("De-Vignette", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Sigma", "De-Vignette", sigma, 3000, change_sigma)

    while True:
        devignetted = ic.de_vignette(linearized_image, sigma)
        cv2.imshow("De-Vignette", devignetted)
        key = cv2.waitKey(1)
        if key == 27:
            break

    crop = crop_image(devignetted)
    crop = resize(crop, 2)

    thresh = ic.thresholding(crop)

    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    cv2.drawContours(crop, contours, -1, (0, 255, 0), 2)
    print(contours)
    cv2.imshow("Thresholding", thresh)
    cv2.imshow("Contours", crop)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

# automatically draw bounding box around sample
