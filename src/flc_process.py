#! /usr/bin/env python

import rospy
import numpy as np
import time
# import actionlib
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
# from cola2_msgs.msg import NavSts

# from process_and_track import process_and_track #Vered's algo
from image_denoise import image_denoise  #Yevgeni denoising
# from dock_detection import dock_detection# find segments
# from target_tracking import target_tracking# track the target

import tf
#for topic synchronization
import message_filters

#for rviz markers
# from visualization_msgs.msg import Marker
# from visualization_msgs.msg import MarkerArray


# Globals
br = CvBridge()


class FlcProcess(object):

    def __init__(self):
        """Constructor that gets config, publishers and subscribers."""

        self.ns = "/sparus2"

        # camera_info_subs = rospy.Subscriber("/camera/camera_info",CameraInfo,self.camera_info_callback)

        #camera_image_subs= rospy.Subscriber(self.ns + "camera/image_color/compressed", CompressedImage, self.camera_mask)
        
        self.pub_camera_mask = rospy.Publisher(self.ns + "camera/object_mask/compressed", CompressedImage, queue_size=10)
        
        
        camera_image_subs= rospy.Subscriber("/camera/image_raw", Image, self.camera_mask)




    def camera_mask(self, camera_image):
        #start = time.time()

        r = 0.3 # resize ratio
        
         #print("camera cb")
        cv_image = br.imgmsg_to_cv2(camera_image)

       # print("Time after cv_image : " + str(time.time() - start))

        h = cv_image.shape[1]
        w = cv_image.shape[0]  

        #print(h,w)
        #w , h  = camera_image.size

        # resize the image
        cv_image = cv2.resize(cv_image, ( int(h*r),  int(w*r)), interpolation = cv2.INTER_AREA)


        gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BayerGR2GRAY)
        #print("Time after gray_img : " + str(time.time() - start))

        gray_blured = cv2.GaussianBlur(gray_img, (7, 7), 0)
        #print("Time after gray_blured " + str(time.time() - start))

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(7,7))
        cl1 = clahe.apply(gray_blured)
        #print("Time after cl1 : " + str(time.time() - start))

        thresholded = cv2.adaptiveThreshold(cl1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        #print("Time after thresholded : " + str(time.time() - start))

        thresholded= cv2.medianBlur(thresholded, 9)
        #print("Time after thresholded with medianBlur : " + str(time.time() - start))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        close  = cv2.morphologyEx(np.invert(thresholded), cv2.MORPH_CLOSE,kernel)
        Ex  = cv2.morphologyEx(close, cv2.MORPH_GRADIENT,kernel)
        #print("Time after Ex : " + str(time.time() - start))

        mask = Ex 
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print("Time after contours : " + str(time.time() - start))


        # Define the top-left and bottom-right coordinates of the rectangle
        top_left =   (0, int(h*r)-300)  # (x, y) coordinates of the top-left corner
        bottom_right = (int(w*r)+200, int(h*r))  # (x, y) coordinates of the bottom-right corner

        # Define the color of the rectangle (in BGR format)
        color = (0, 0, 0)  # Black color

        # Draw a black rectangle on the image
        mask = cv2.rectangle(mask, top_left, bottom_right, color, thickness=cv2.FILLED)

        top_left =   (int(w*r)+30, int(h*r)-450)  # (x, y) coordinates of the top-left corner
        bottom_right = (int(w*r)+300, int(h*r))  # (x, y) coordinates of the bottom-right corner

        # Draw a black rectangle on the image
        mask = cv2.rectangle(mask, top_left, bottom_right, color, thickness=cv2.FILLED)

        top_left =   (int(w*r)+130, int(h*r)-500)  # (x, y) coordinates of the top-left corner
        bottom_right = (int(w*r)+500, int(h*r))  # (x, y) coordinates of the bottom-right corner

        # Draw a black rectangle on the image
        mask = cv2.rectangle(mask, top_left, bottom_right, color, thickness=cv2.FILLED)

        # num_labels, labels = cv2.connectedComponents(mask)
        # label_hue = np.uint8(179*labels/np.max(labels))
        # blank_ch = 255*np.ones_like(label_hue)
        # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # #print("Time after labeled_img : " + str(time.time() - start))

        # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        # labeled_img[label_hue==0] = 0
        #print("Time after final labeled_img : " + str(time.time() - start))

        #mask = cv2.resize(mask, (int(h/r), int(w/r)), interpolation = cv2.INTER_AREA)


        mask_img = br.cv2_to_imgmsg(mask)

        msg = CompressedImage()
        msg.header.stamp = camera_image.header.stamp #rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', mask)[1]).tostring()

        self.pub_camera_mask.publish(msg)
        #print("Time after msg: " + str(time.time() - start))
        

  

if __name__ == '__main__':
    # init
    rospy.init_node('flc_process')
    node = FlcProcess()

    r = rospy.Rate(100) # 10hz 
    while not rospy.is_shutdown():
        # pub.publish("hello")
        # node.loop()
        r.sleep()

    # rospy.spin()ll