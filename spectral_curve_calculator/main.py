import numpy as np
import image_corrections as ic
import cv2
from tkinter import Tk, filedialog

sigma = 900
thresh_val = 40


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


class Sample:
    def __init__(self, name):
        self.name = name
        self.image = open_image()
        self.linearized = ic.gamma_expand(self.image)

    def resize(self, scale):
        width = int(self.image.shape[1] * scale)
        height = int(self.image.shape[0] * scale)
        dim = (width, height)
        self.image = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)
        self.linearized = cv2.resize(self.linearized, dim, interpolation=cv2.INTER_AREA)

    def display(self):
        cv2.imshow(self.name, self.linearized)
        cv2.waitKey(0)


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


def crop_image(image):
    r = cv2.selectROI(image)
    cropped = image[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
    return cropped


def main():
    sample = Sample('sample 1')
    sample.resize(0.3)

    # learn to auto-adjust brightness
    linearized_image = sample.linearized

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
    print(crop.shape)
    thresh = ic.thresholding(crop)

    contour_mask = np.zeros(crop.shape[:2], np.uint8)
    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    cv2.drawContours(contour_mask, contours, -1, 255, 2)

    # avg color in BGR not RGB
    avgColor = cv2.mean(crop, contour_mask)
    print(contours)
    print(avgColor)
    cv2.imshow("Thresholding", thresh)
    cv2.imshow("Contours", contour_mask)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

# automatically draw bounding box around sample
