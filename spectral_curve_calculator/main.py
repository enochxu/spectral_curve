import numpy as np
import matplotlib.pyplot as plt
import image_corrections as ic
import cv2
from tkinter import Tk, filedialog

# 778
# 2592
sigma = 2592
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


def crop_image(image):
    cv2.namedWindow("Select ROI", cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Select ROI', 800, 600)
    r = cv2.selectROI('Select ROI', image)
    cropped = image[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
    return cropped


def main():
    sample = Sample('sample 1')
    # s

    # learn to auto-adjust brightness
    linearized_image = sample.linearized
    image = sample.image

    # cv2.imshow("BW", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    # key = cv2.waitKey(1)

    # plt.show()

    global sigma
    # cv2.namedWindow("De-Vignette", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("De-Vignette", cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('De-Vignette', 800, 600)
    cv2.createTrackbar("Sigma", "De-Vignette", sigma, 3000, change_sigma)

    while True:
        devignetted = ic.de_vignette(linearized_image, sigma)
        cv2.imshow("De-Vignette", devignetted)
        key = cv2.waitKey(1)
        if key == 27:
            break

    filename = 'linearizedImage.png'
    cv2.imwrite('/samples/linearizedImage.png', devignetted)
    crop = crop_image(devignetted)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (19, 19), 0)

    fig0 = plt.figure(0, figsize = (8,6))
    ax0 = fig0.gca()
    ax0.set_xlabel('Pixel Intensity')
    ax0.set_ylabel('Pixel Count')
    plt.hist(blurred.ravel(), 256, [0, 256])
    plt.show()

    retval, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fig1 = plt.figure(1, figsize = (8,6))
    ax1 = fig1.gca()
    ax1.set_xlabel('Pixels(px)')
    ax1.set_ylabel('Pixels(px)')
    plt.imshow(th, cmap='gray')
    plt.show()

    fig2 = plt.figure(1, figsize=(8, 6))
    ax2 = fig2.gca()
    ax2.set_xlabel('Pixels(px)')
    ax2.set_ylabel('Pixels(px)')
    # plt.imshow(cv2.cvtColor(linearized_image, cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(devignetted, cv2.COLOR_BGR2GRAY), cmap='gray')
    # plt.show()
    plt.imshow(gray, cmap='gray')
    plt.show()
    fig3 = plt.figure(1, figsize = (8,6))
    ax3 = fig3.gca()
    ax3.set_xlabel('Pixels(px)')
    ax3.set_ylabel('Pixels(px)')
    plt.imshow(blurred, cmap='gray')
    plt.show()
    # global thresh_val
    # cv2.namedWindow("Thresholding", cv2.WINDOW_AUTOSIZE)
    # cv2.createTrackbar("Thresh Value", "Thresholding", thresh_val, 200, change_thresh)

    contour_mask = np.zeros(th.shape, np.uint8)
    # kernel = np.ones((15, 15), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(9, 9)) # 9,9
    erode = cv2.morphologyEx(th, cv2.MORPH_ERODE, kernel)
    fig3 = plt.figure(3, figsize = (8,6))
    ax3 = fig3.gca()
    ax3.set_xlabel('Pixels(px)')
    ax3.set_ylabel('Pixels(px)')
    plt.imshow(blurred, cmap='gray')
    plt.imshow(erode, cmap='gray')
    plt.show()
    dilate = cv2.morphologyEx(th, cv2.MORPH_DILATE, kernel)
    fig3 = plt.figure(1, figsize = (8,6))
    ax3 = fig3.gca()
    ax3.set_xlabel('Pixels(px)')
    ax3.set_ylabel('Pixels(px)')
    plt.imshow(blurred, cmap='gray')
    plt.imshow(dilate, cmap='gray')
    plt.show()
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    fig3 = plt.figure(1, figsize = (8,6))
    ax3 = fig3.gca()
    ax3.set_xlabel('Pixels(px)')
    ax3.set_ylabel('Pixels(px)')
    plt.imshow(blurred, cmap='gray')
    plt.imshow(opening, cmap='gray')
    plt.show()
    contours, hierarchy = cv2.findContours(
        opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    cv2.drawContours(contour_mask, contours, -1, 255, 2)
    print(cv2.mean(crop, contour_mask))
    cv2.drawContours(crop, contours, -1, 255, 2)
    fig4 = plt.figure(1, figsize = (8,6))
    ax4 = fig4.gca()
    ax4.set_xlabel('Pixels(px)')
    ax4.set_ylabel('Pixels(px)')
    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imshow("Contours", crop)
    cv2.waitKey(0)

    sample = crop_image(crop)
    print(cv2.mean(sample))
    # while True:
    #     thresholded = ic.thresholding(crop, thresh_val)
    #     cv2.imshow("Thresholding", thresholded)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break

    thresh = ic.thresholding(crop)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)
    contour_mask = ic.contour_mask(thresh)

    # avg color in BGR
    avgColor = cv2.mean(crop, contour_mask)
    # print(contours)
    # print(avgColor)
    # cv2.imshow("Thresholding", thresh)
    # cv2.imshow("Contours", contour_mask)
    # cv2.waitKey(0)


if __name__ == "__main__":
    main()

# automatically draw bounding box around sample
