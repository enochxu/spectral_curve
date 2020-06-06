import numpy as np
import matplotlib.pyplot as plt
import ctools
from cv2 import cv2
from tkinter import Tk
from tkinter import filedialog

sigma = 900
thresh_val = 40

def change_sigma(new_sigma):
    global sigma 
    sigma = new_sigma

def change_tresh(new_thresh):
    global thresh_val
    thresh_val = new_thresh

def resize(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#let user choose file, returns image
def open_image():
    root = Tk()
    root.fileName = filedialog.askopenfilename(initialdir='./samples', 
                                                title='Select an Image', 
                                                filetypes=(('.jpg', '*.jpg'), 
                                                ('.png', '*.png'), ('all files', '*.*')))                                        
    fname = root.fileName
    image = cv2.imread(fname)
    return image

def crop_image(image):
    r = cv2.selectROI(image)
    cropped = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    return cropped

#method creates vignette with a gaussian kernel
def de_vignette(image, sigma):
    #setting the gaussian deviation to match the size of the image
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, sigma)
    kernel_y = cv2.getGaussianKernel(rows, sigma)
    #sigma = gaussian standard deviation

    kernel = kernel_y * kernel_x.T
    mask = kernel/kernel.max()

    vignette = np.zeros_like(image)
    for i in range(3):
        vignette[:,:,i] = image[:,:,i] * (1 / mask)
    return vignette

def thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    retval, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def main():
    substrate_color = [95.65867622736137, 41.89439137669053, 77.11017914916165]
    monolayer_graphene_color = [92.09239342300415, 36.123467574326696, 75.95457000069112]

    image = open_image()
    linearized_image = ctools.gamma_expand(image)
    #resize image
    image = resize(image, 0.3)
    linearized_image = resize(linearized_image, 0.3)
    #learn to auto-adjust brightness
    
    global sigma
    cv2.namedWindow("De-Vignette", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Sigma', 'De-Vignette', sigma, 3000, change_sigma)

    while True:
        devignetted = de_vignette(linearized_image, sigma)
        cv2.imshow('De-Vignette', devignetted)
        key = cv2.waitKey(1)
        if key == 27:
            break

    crop = crop_image(devignetted)
    crop = resize(crop, 2)
    
    thresh = thresholding(crop)

    kernel = np.ones((15,15), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(crop, contours, -1, (0, 255, 0), 2)   
    print(contours)
    cv2.imshow('Thresholding', thresh)
    cv2.imshow('Contours', crop)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

    # gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # cv2.imshow('original', crop)
    # cv2.waitKey(0)

    # global thresh_val
    # cv2.namedWindow("Thresholding", cv2.WINDOW_AUTOSIZE)
    # cv2.createTrackbar('Threshold', 'Thresholding', thresh_val, 255, change_tresh)
    # kernel = np.ones((15,15), np.uint8)
    # while True:
    #     retval, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #     # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    #     closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #     contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cv2.drawContours(crop, contours, -1, (0, 255, 0), 2)
    #     images = np.hstack((blurred, thresh, closing))
    #     cv2.imshow('Thresholding', images)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break

    # cv2.imshow('vignette', de_vignette(linearized_image, 900))


    # cv2.namedWindow("Images", cv2.WINDOW_AUTOSIZE)
    # cv2.createTrackbar('Low Threshold', 'Images', 0, 255, nothing)
    # cv2.createTrackbar('High Threshold', 'Images', 0, 255, nothing)



    # while True:
    #     # filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    #     #filtering to minimize noise
    #     low = cv2.getTrackbarPos('Low Threshold', 'Images')
    #     high = cv2.getTrackbarPos('High Threshold', 'Images')         

    #     images = canny_edge(crop, low, high)
    #     cv2.imshow('Images', images)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break

    # _, thresh = cv2.threshold(blurred,120,255,cv2.THRESH_BINARY)
    # cv2.imshow('gray', resize(edges, 20))
    # image = resize(image, 20)
    # cropped = crop_image(image)
    # cv2.imshow('cropped', resize(cropped,500))
    # cv2.imshow('image', resize(image, 20))
    # cv2.imshow('Linearized image', resize(linearized_image, 20))
    # def canny_edge(image, low, high):
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    #     # filtered = cv2.bilateralFilter(gray, 7, 50, 50)

    #     edges = cv2.Canny(blurred, low, high)
    #     return edges


#automatically draw bounding box around sample

