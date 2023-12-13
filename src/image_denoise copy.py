import cv2
import numpy as np
import objDetect
import imutils
#import particle_filter
#import kalman_filter
import mixture_particles_filter
from visualize import show_particles, show_objects
from help_functions import find_center
from get_objects import get_objects
import matplotlib.pyplot as plt
from identification import detect_target2

image_prev =  np.zeros((100,100,3), dtype=np.uint8) #[0]

class image_denoise:


    def __init__(self):

        self.threshhold_max = 120  
        self.threshhold_min = 30  
        self.max_pixel_num  = 500 

        self.image_prev = image_prev
        self.dx   = 0
        self.dy   = 0
        self.dtheta   = 0

        # find mask file
        import os
        import rospkg
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("fls_detection")
        mask = cv2.imread(os.path.join(package_path, "mask", "mask.jpg"))
        self.beams_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY )   

    def image_denoise(self, img, range_img=None):
        self.o = objDetect.ROIfind(img.copy())
        mask = self.o.create_rois_map(img.copy(), range_img)
        self.beams_denoise(img)
        return mask


    def image_denoise_2(self, img):

        #img = np.invert(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))


        #rot = cv2.getRotationMatrix2D((cX, cY), dtheta, 1.0)
        #rotated = cv2.warpAffine(self.image_prev, rot, (self.image_prev.shape[1], self.image_prev.shape[0]))

        #trans = np.float32([[1, 0, -dy],[0, 1, -dx]])
        #shifted = cv2.warpAffine(rotated, trans, (self.image_prev.shape[1], self.image_prev.shape[0]))


        if self.image_prev.shape == img.shape:

            #img = (img + self.image_prev) / 2

              img = cv2.addWeighted(img,0.7,self.image_prev,0.3,0)

            #print(np.amax(self.image_prev))



        #blurred = cv2.GaussianBlur(img, (3, 3), 0)
        blurred = cv2.medianBlur(img, 3)


        #Compute histogram
        hist = cv2.calcHist([img], [0], None, [self.threshhold_max], [self.threshhold_min, self.threshhold_max])

        #Convert histogram to simple list
        hist = [val[0] for val in hist]

        #Generate a list of indices
        indices = list(range(self.threshhold_min, self.threshhold_max))

        #Descending sort-by-key with histogram value as key
        s = [(x,y) for y,x in sorted(zip(hist,indices), reverse=True)]
    

        position = [i for i, tupl in enumerate(s) if (tupl[1] < self.max_pixel_num)]

        min_th = (s[position[1]][0]) 

        #print(index_of_highest_peak)
        #print(index_of_second_highest_peak)

        #plt.hist(img.ravel(),256,[2,256]); plt.show()

        # #If top 2 indices are adjacent to each other, there won't be a midpoint
        # if abs(index_of_highest_peak - index_of_second_highest_peak) < 2:
        #     raise Exception('Midpoint does not exist')
        # else: #Compute mid index
        #     midpoint = int( (index_of_highest_peak + index_of_second_highest_peak) / 2.0 )



        # print('Index Between Top 2 Peaks = ', midpoint)
        # print('Histogram Value At MidPoint = ', hist[midpoint])



        #thresholded = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,3)
        ret,thresholded = cv2.threshold(blurred,min_th,255,cv2.THRESH_BINARY)
        #thresholded = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,3,3)
        #thresholded = np.invert(thresholded)
        #otsu_threshold, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #image_result = np.invert(image_result)
        close  = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE,kernel)


        #grad = cv2.morphologyEx(close, cv2.MORPH_GRADIENT, kernel)
        #dilation = cv2.dilate(thresholded,kernel,iterations = 1)
        #open  = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN,kernel)
        # open = np.invert(open)
        # #image_result = np.invert(image_result)
        #th       = cv2.morphologyEx(thresholded, cv2.MORPH_TOPHAT,kernel)
        #blurred2 = cv2.GaussianBlur(th, (5, 5), 0)
        #mg       = cv2.morphologyEx(close, cv2.MORPH_GRADIENT,kernel)

        #clustered = self.clustering(close)

        #segmented = self.denoise_segments(close)
        #edged = cv2.Canny(img, 50, 100)
        #edged = cv2.dilate(edged, None, iterations=1)
        #edged = cv2.erode(edged, None, iterations=1)
        #segmented = self.denoise_segments(close)

        self.image_prev = cv2.medianBlur(img, 5) 
        #print(np.amax(self.image_prev))

        return close


    def beams_denoise(self, img):

        h , w = img.shape

        resized_mask = cv2.resize(self.beams_mask, (w , h), interpolation = cv2.INTER_AREA)
 
        masked_img = img - 0 * resized_mask

        #cv2.imshow("mask", masked_img)
        #cv2.waitKey(0)
        
        return masked_img
