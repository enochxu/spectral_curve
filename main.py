import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
from tkinter import Tk
from tkinter import filedialog


def main():
    root = Tk()
    root.fileName = filedialog.askopenfilename(initialdir="/samples", title="Select an Image", 
                                            filetypes=((".jpg", "*.jpg"),(".png", "*.png"),("all files", "*.*")))
    fpath = root.fileName
    image = cv2.imread(fpath, 1)


if __name__ == '__main__':
    main()



