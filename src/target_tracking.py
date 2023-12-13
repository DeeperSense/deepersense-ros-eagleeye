import cv2
import numpy as np

import matplotlib.pyplot as plt



class target_tracking:

    def __init__(self):


        # Change thresholds
        self.minThreshold = 50

    def sonarToworld(self, bboxes):
        self.minThreshold = 4

    def bboxes_track(self, bboxes):
        self.minThreshold = 4    
