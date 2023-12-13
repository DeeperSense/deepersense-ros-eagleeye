import cv2
import numpy as np
import matplotlib.pyplot as plt
from help_functions import window_smoothing, find_center

# class for pre-processing of the image
class ROIfind:
    def __init__(self, img):
        self.img = img
        self.i = 1
        self.evaluate_noise()

    def evaluate_noise(self):

        # cv2.imshow('img', self.img)
        #blurred_image = cv2.GaussianBlur(self.img,(9,9),0)
        hist = cv2.calcHist([self.img], [0], None, [256], [0,256]) / np.product(self.img.shape)
        b_ground = 5 #5 absolute background threshold

        # find the point seperate noise and signal:
        local_max_idx = np.argmax(hist[b_ground:150]) + b_ground
        local_max = np.max(hist[b_ground:150])
        local_min_idx = np.argmin(hist[local_max_idx:200]) + local_max_idx
        local_min = np.min(hist[local_max_idx:200])
        one_third_noise = (local_max - local_min) / 3
        self.noise_end = local_max_idx + np.argmin(hist[local_max_idx:local_min_idx] - one_third_noise)


        # b_ground = np.argmin(hist[b_ground:local_max_idx])+b_ground if local_max_idx > b_ground else local_max_idx
        noise = np.sum(hist[b_ground:self.noise_end])
        signal = np.sum(hist[self.noise_end:]) #+ np.sum(hist[1:b_ground])  # absolute highlights + absolute background
        self.snr = np.min([1, signal/noise]) # creates good estimation for pool/sea...
        #print(self.snr)
        if self.snr > 0.1:
            self.img = cv2.blur(cv2.equalizeHist(self.img), (5,5))
            hist = cv2.calcHist([self.img], [0], None, [256], [0,256])
            # find the point seperate noise and signal:
            local_max_idx = np.argmax(hist[b_ground:150])+b_ground
            local_max = np.max(hist[b_ground:150])
            local_min_idx = np.argmin(hist[local_max_idx:200]) + local_max_idx
            local_min = np.min(hist[local_max_idx:200])
            one_third_noise = (local_max - local_min) / 3
            self.noise_end = local_max_idx + np.argmin(hist[local_max_idx:local_min_idx] - one_third_noise)

        #self.noise_end = otsu_threshold

    # create map of regions-of-interest from the image
    def create_rois_map(self, img, range_img):
        self.img = img
        #erase noise close to the sonar:
        self.img[np.where(range_img < 0.5)] = 0

        self.evaluate_noise()

        """ if self.snr > 0.15:
            #echo = np.max([int(self.snr * 30), 10])  # echo_size is proportional to the image quality, or mnimum 10
            # create echo map using window smoothing
            echo_map = window_smoothing(self.img, 30)
            echo_map = cv2.morphologyEx(echo_map, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        else:
            echo_map = window_smoothing(self.img, 30) - window_smoothing(self.img, 15)
            # echo_map = self.img

        # _, rois_map = cv2.threshold(echo_map, self.noise_end, 255, cv2.THRESH_BINARY)

        # dilating the image rois:
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
        #rois_map = cv2.dilate(rois_map, kernel) """

        rois_map = self.img
        rois_map[np.where(rois_map<self.noise_end)] = 0

        if self.snr > 0.15:
            echo_map = window_smoothing(rois_map, 30) - window_smoothing(rois_map, 15)
            echo_map[np.where(echo_map<np.median(np.unique(echo_map)))]=0
            echo_map[np.where(echo_map>np.median(np.unique(echo_map)))] = self.img[np.where(echo_map>np.median(np.unique(echo_map)))]
            echo_map[np.where(echo_map<self.noise_end)]=0
            rois_map = cv2.morphologyEx(echo_map, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            rois_map[np.where(rois_map<self.noise_end)] = 0

        #plt.imshow(rois_map)
        #plt.show()

        return rois_map
