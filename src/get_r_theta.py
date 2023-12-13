import numpy as np
from help_functions import find_center
from scipy.ndimage import binary_fill_holes
import cv2
import os
import math

class get_r_theta:


    def get_r_theta(self, img, ranges):

        center = find_center(img)
        hight, width = img.shape
        rhight, rwidth = ranges.shape
        theta_img = np.zeros((hight, width))
        coordinates = np.where(ranges > [0.5])
        theta = np.arctan2(coordinates[1] - center[0], center[1] - coordinates[0])
        theta_img[coordinates] = theta #+ 2 * math.pi 


        #print([hight, width])
        #print([rhight, rwidth])
        #print(theta)
        #print(zip(coordinates[0], coordinates[1]))
        #print(theta_img[:,center[0]])
        #print(np.max(ranges))
        #print(theta_img[:,center[1]])
        return theta_img
