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
import rospkg
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


image_prev =  np.zeros((100,100,3), dtype=np.uint8) #[0]

class image_denoise:


    def __init__(self):

        self.threshhold_max = 40  
        self.threshhold_min = 20  
        self.max_pixel_num  = 100 

        self.image_prev = image_prev
        self.dx   = 0
        self.dy   = 0
        self.dtheta   = 0

        self.gamma = 0.5
        self.gamma_inv = 1.0 / self.gamma
        self.gamma_table = np.array([((i / 255.0) ** self.gamma_inv) * 255  for i in np.arange(0, 256)]).astype("uint8")


        self.adaptive_low_threshold = 50
        self.adaptive_high_threshold = 220
        self.min_threshold_offset = -10

        


        self.bridge = CvBridge()
        self.pub_image_debug0 = rospy.Publisher('/image_debug0', Image, queue_size=1)
        self.pub_image_debug1 = rospy.Publisher('/image_debug1', Image, queue_size=1)
        self.pub_image_debug2 = rospy.Publisher('/image_debug2', Image, queue_size=1)
        self.pub_image_debug3 = rospy.Publisher('/image_debug3', Image, queue_size=1)
        self.pub_image_debug4 = rospy.Publisher('/image_debug4', Image, queue_size=1)
        self.pub_image_debug5 = rospy.Publisher('/image_debug5', Image, queue_size=1)
        self.pub_image_debug6 = rospy.Publisher('/image_debug6', Image, queue_size=1)
        self.pub_image_debug7 = rospy.Publisher('/image_debug7', Image, queue_size=1)
        self.pub_image_debug8 = rospy.Publisher('/image_debug8', Image, queue_size=1)
        self.pub_image_debug9 = rospy.Publisher('/image_debug9', Image, queue_size=1)
        self.pub_image_debug10 = rospy.Publisher('/image_debug10', Image, queue_size=1)

        # get mask file path from package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('fls_detection')
        #mask = cv2.imread(os.path.join(package_path, 'mask' ,'mask.jpg'))
        #self.beams_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY )   




    def reconfigure_callback(self, config):
        self.threshhold_max = config['threshold_max']  
        self.threshhold_min = config['threshold_min']
        self.max_pixel_num  = config['max_pixel_num']
        
        # Gamma correction
        self.enable_gamma_correction = config['enable_gamma_correction']
        self.gamma = config['gamma']
        self.gamma_inv = 1.0 / self.gamma
        self.gamma_table = np.array([((i / 255.0) ** self.gamma_inv) * 255  for i in np.arange(0, 256)]).astype("uint8")

        # Adaptive beam denoising
        self.enable_adaptive_beam_denoising = config['enable_adaptive_beam_denoising']
        self.adaptive_low_threshold = config['adaptive_low_threshold']
        self.adaptive_high_threshold = config['adaptive_high_threshold']
        self.min_threshold_offset = config['min_threshold_offset']

        # Warping
        self.enable_warping = config['enable_warping']
        self.warping_mixing_factor = config['warping_mixing_factor']

        # Blur
        self.enable_blur = config['enable_blur']
        self.blur_kernel_size = config['blur_kernel_size']

        # Morphological operations
        self.enable_morphological_operations = config['enable_morphological_operations']

        return config

    def image_denoise(self, img, range_img=None):
        self.o = objDetect.ROIfind(img.copy())
        mask = self.o.create_rois_map(img.copy(), range_img)
        self.beams_denoise(img)
        return mask


    def image_denoise_2(self, img,dx,dy,dpsi, origin_col, origin_row):

        # For Stonefish or rgb image
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.pub_image_debug0.publish(self.bridge.cv2_to_imgmsg(img, "mono8"))

        # gamma correction
        if self.enable_gamma_correction:
            img = cv2.LUT(img, self.gamma_table)
        self.pub_image_debug1.publish(self.bridge.cv2_to_imgmsg(img, "mono8"))


        dx = int(dx)
        dy = int(dy)
        center = (origin_col, origin_row)
        height, width = img.shape[:2]


        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=dpsi, scale=1)
        translation_matrix = np.array([[1, 0, dx],[0, 1, dy]], dtype=np.float32)
        #img = np.invert(img)


        masked_img = self.adaptive_beams_denoise(img, origin_col, origin_row)
        self.pub_image_debug2.publish(self.bridge.cv2_to_imgmsg(masked_img, "mono8"))
        # cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        # cv2.resizeWindow("output", 400, 300)  
        # cv2.imshow("output", masked_img)
        # cv2.waitKey(0)
        if self.enable_adaptive_beam_denoising:
            img = cv2.bitwise_and(img,img,mask = masked_img)


        self.pub_image_debug3.publish(self.bridge.cv2_to_imgmsg(img, "mono8"))

        #img = cv2.bitwise_and(img,img,mask = masked_img)
        #rot = cv2.getRotationMatrix2D((cX, cY), dtheta, 1.0)
        #rotated = cv2.warpAffine(self.image_prev, rot, (self.image_prev.shape[1], self.image_prev.shape[0]))

        #trans = np.float32([[1, 0, -dy],[0, 1, -dx]])
        #shifted = cv2.warpAffine(rotated, trans, (self.image_prev.shape[1], self.image_prev.shape[0]))


        if self.image_prev.shape == img.shape:

            translated_image = cv2.warpAffine(src=self.image_prev, M=translation_matrix, dsize=(width, height))
            self.pub_image_debug4.publish(self.bridge.cv2_to_imgmsg(translated_image, "mono8"))

            rotated_image = cv2.warpAffine(src=translated_image, M=rotate_matrix, dsize=(width, height))


            #img = cv2.addWeighted(img,1.0 - self.warping_mixing_factor,rotated_image,self.warping_mixing_factor,0)
            img = cv2.addWeighted(img,0.5,rotated_image,0.5,0)

            self.pub_image_debug5.publish(self.bridge.cv2_to_imgmsg(img, "mono8"))

            # cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
            # cv2.resizeWindow("output", 400, 300)  
            # cv2.imshow("output", img)
            # cv2.waitKey(0)

            #print(np.amax(self.image_prev))



        #blurred = cv2.GaussianBlur(img, (3, 3), 0)

        if self.enable_blur:
            # kernel size should be odd
            kernel_size = self.blur_kernel_size if self.blur_kernel_size % 2 == 1 else self.blur_kernel_size + 1
            blurred = cv2.medianBlur(img, kernel_size)
            # blurred = cv2.medianBlur(img, 4)
        else:
            blurred = img

        self.pub_image_debug6.publish(self.bridge.cv2_to_imgmsg(blurred, "mono8"))


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

        # plt.hist(img.ravel(),256,[2,256]); plt.show()

        # #If top 2 indices are adjacent to each other, there won't be a midpoint
        # if abs(index_of_highest_peak - index_of_second_highest_peak) < 2:
        #     raise Exception('Midpoint does not exist')
        # else: #Compute mid index
        #     midpoint = int( (index_of_highest_peak + index_of_second_highest_peak) / 2.0 )



        # print('Index Between Top 2 Peaks = ', midpoint)
        # print('Histogram Value At MidPoint = ', hist[midpoint])



        #thresholded = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,3)
        ret,thresholded = cv2.threshold(blurred,min_th + self.min_threshold_offset ,255,cv2.THRESH_BINARY)
        self.pub_image_debug7.publish(self.bridge.cv2_to_imgmsg(thresholded, "mono8"))

        #thresholded = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,3,3)
        #thresholded = np.invert(thresholded)
        #otsu_threshold, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #image_result = np.invert(image_result)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        if self.enable_morphological_operations:
            close  = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE,kernel)
        else:
            close = thresholded


        # close  = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE,kernel)


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

        # self.image_prev = cv2.medianBlur(img, 5) 





        self.image_prev = blurred

        #print(np.amax(self.image_prev))
        #return  masked_img


        # cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        # cv2.resizeWindow("output", 400, 300)  
        # cv2.imshow("output", close)
        # cv2.waitKey(0)
        #close = close 
        #cv2.line(close,(0,50),(width,50),(255,0,0),10)

        # return masked_img
        return close




    def image_denoise_oculus(self, img,dx,dy,dpsi, origin_col, origin_row):


        img = cv2.flip(img, 1)
        dx = int(dx)
        dy = int(dy)
        center = (origin_col, origin_row)
        height, width = img.shape[:2]


        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=dpsi, scale=1)
        translation_matrix = np.array([[1, 0, dx],[0, 1, dy]], dtype=np.float32)

        blurred = cv2.medianBlur(img,  3)



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

        #print(min_th)
        #thresholded = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,3)
        ret,thresholded = cv2.threshold(blurred,min_th+30 ,255,cv2.THRESH_BINARY)
        #ret,thresholded = cv2.threshold(blurred,80 ,255,cv2.THRESH_BINARY)
        #self.pub_image_debug7.publish(self.bridge.cv2_to_imgmsg(thresholded, "mono8"))

        #thresholded = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,3,3)
        #thresholded = np.invert(thresholded)
        #otsu_threshold, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #image_result = np.invert(image_result)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

        close  = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE,kernel)

        # close  = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE,kernel)


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

        # self.image_prev = cv2.medianBlur(img, 5) 

        # mask the middle
        rect_color = (0, 0, 0)  # Rectangle color in BGR format (green in this case)
        rect_thickness = 6  # Thickness of the rectangle border
        rect_width = 8  # Width of the rectangle
        rect_height = height  # Height of the rectangle

        # Calculate the coordinates to draw the rectangle in the middle of the image
        x = (width - rect_width) // 2
        y = (height - rect_height) // 2
        end_x = x + rect_width
        end_y = y + rect_height

        close = cv2.rectangle(close, (x, y), (end_x, end_y), rect_color, thickness=cv2.FILLED)


        close = cv2.rectangle(close, (0, height), (width, height-10), rect_color, thickness=cv2.FILLED)


        self.image_prev = blurred

        #print(np.amax(self.image_prev))
        #return  masked_img


        # cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        # cv2.resizeWindow("output", 400, 300)  
        # cv2.imshow("output", close)
        # cv2.waitKey(0)
        #close = close 
        #cv2.line(close,(0,50),(width,50),(255,0,0),10)

        # return masked_img
        return close



    def beams_denoise(self, img):

        h , w = img.shape

        resized_mask = cv2.resize(self.beams_mask, (w , h), interpolation = cv2.INTER_AREA)
 
        masked_img = img - 1 * resized_mask

        #cv2.imshow("mask", masked_img)
        #cv2.waitKey(0)
        
        return masked_img

    def adaptive_beams_denoise(self, img, origin_col, origin_row):

        # self.adaptive_low_threshold = 240
        # self.adaptive_high_threshold = 255
        edges = cv2.Canny(img, self.adaptive_low_threshold, self.adaptive_high_threshold)
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 100  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 30  # minimum number of pixels making up a line
        max_line_gap = 1000  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 255  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    if abs(origin_col - x1) > 1  and abs(x2 - x1) > 1 and abs(y2 - y1) > 1:
                        if 0.9 <=( ((origin_row - y1) / (origin_col - x1) ) / ((y2 - y1) / (x2 - x1) )) <=1.2:

                            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,0),10)


    # add a black circle to the mask to remove the robot
        # get the origin of the robot from the center of the image
        origin_x = int(line_image.shape[1]/2)
        origin_y = int(line_image.shape[0])
        # get the radius of the robot
        radius = int(line_image.shape[0]/2)
        scale = 0.2
        # draw the circle
        #line_image = cv2.circle(line_image, (origin_x, origin_y), int(radius * scale), (0,0,0), -1)




        # cv2.namedWindow("line_image", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        # cv2.resizeWindow("line_image", 400, 300)
        # cv2.imshow("line_image", line_image)
        # cv2.waitKey(0)    
        
        return line_image
    


