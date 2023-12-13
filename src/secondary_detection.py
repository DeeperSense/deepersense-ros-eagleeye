import cv2
import numpy as np
import objDetect
#import particle_filter
#import kalman_filter
import mixture_particles_filter

from get_objects import get_objects
import matplotlib.pyplot as plt



class secondary_detection:

    def __init__(self):

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 50
        params.maxThreshold = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 500
        params.maxArea = 1000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.3

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            self.detector = cv2.SimpleBlobDetector(params)
        else : 
            self.detector = cv2.SimpleBlobDetector_create(params)

        #self.detector = cv2.SimpleBlobDetector_create(params)

    def find_target_bulb(self,img):
        #self.img = img
        # Detect blobs.
        keypoints = self.detector.detect(img)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        #cv2.imshow("Keypoints", im_with_keypoints)
        #cv2.waitKey(500)
        #cv2.destroyAllWindows()



    # def secondary_detection(self):



    #     self.find_target_bulb(self)

    #     target = 1
    #     return target


