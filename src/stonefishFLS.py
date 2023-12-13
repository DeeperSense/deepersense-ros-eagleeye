#! /usr/bin/env python

from __future__ import print_function
import sys
import os

import rospy
import numpy as np
import actionlib
import cv2
import struct

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from teledyne_m900_fls.msg import sonar_info

from std_msgs.msg import Header
import rospkg
import xml.etree.ElementTree as ET




# Globals
br = CvBridge()

class StonefishFLS(object):

    def __init__(self):
        """Constructor that gets config, publishers and subscribers."""
        rospack = rospkg.RosPack()
        # Get path to the stonefish sparus2_haifa.scn file
        package_path = rospack.get_path('cola2_stonefish')
        stonefish_settings_file_path = os.path.join(package_path, 'scenarios', 'sparus2_haifa.scn')
        # print('Package path: ' + package_path)
        # get the value for the max range from the stonefish settings file
        tree = ET.parse(stonefish_settings_file_path)
        root = tree.getroot()
        for sensor in root.findall(".//sensor"):
            if sensor.attrib["name"] == "Blue_view_M900" and sensor.attrib["type"] == "fls":
                range_max = sensor.find(".//settings").attrib["range_max"]
                range_max = float(range_max)
                print('Max range from stonefish settings file:', range_max)
        # exit()

        self.min_range = 0.5
        self.max_range = range_max # 50
        self.seq       = 0
        now = rospy.get_rostime()
        # Get namespace and config
        self.ns = rospy.get_namespace()
        rospy.Subscriber('/sparus2/Blue_view_M900_FLS/display', Image,self.image_sub)
        self.pub_son_img  = rospy.Publisher('sparus2/FLS/Img_color/bone/mono', Image, queue_size=100)
        self.pub_son_info = rospy.Publisher('sparus2/FLS/sonar_info', sonar_info, queue_size=100)
        print('Stonefish FLS node initialized')

    def image_sub(self, img_msg):
        
        
        img = br.imgmsg_to_cv2(img_msg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        [xc,  yc, rang_res] = self.params_from_son_img(self.min_range, self.max_range, img)


        height, width    = img.shape
        now              = rospy.get_rostime()

        sonar_msg                   = sonar_info()
        sonar_msg.header                  = Header()
        sonar_msg.file_mode         = False
        sonar_msg.save_to_son       = False
        sonar_msg.son_file_path     = ''
        sonar_msg.son_out_path      = '/home/user/son_files/fls_20220802145534.son'
        sonar_msg.deviceIP          = "192.168.1.45"
        sonar_msg.fls_start_range   = self.min_range
        sonar_msg.fls_stop_range    = self.max_range
        sonar_msg.fls_gamma         = 0.5
        sonar_msg.fls_gain          = 15.0
        sonar_msg.fls_slope         = 0.25
        sonar_msg.fls_sound_speed   = 1500
        sonar_msg.height            = height
        sonar_msg.width             = width
        sonar_msg.origin_row        = yc
        sonar_msg.origin_col        = xc
        sonar_msg.range_resolution  = rang_res
        sonar_msg.header.seq        = self.seq
        #sonar_msg.header.stamp.secs = now.secs
        #sonar_msg.header.stamp.nsecs = now.nsecs
        sonar_msg.header.stamp.secs = img_msg.header.stamp.secs
        sonar_msg.header.stamp.nsecs = img_msg.header.stamp.nsecs
        sonar_msg.header.frame_id   = "sonar"

        img_msg_out = br.cv2_to_imgmsg(img, encoding="passthrough")
        self.pub_son_info.publish(sonar_msg)
        self.pub_son_img.publish(img_msg_out)
        self.seq = self.seq + 1 


    def params_from_son_img(self, min_range, max_range, img):
        # this method find the pcenter of the image: where the FLS is
        rows, cols = img.shape
        lastLine = np.array(img[rows - 1, :])
        l = np.where(lastLine > 0)
        l = l[0] # l is a tuple of list. extract the list.
        # if the last line is empty:
        i = 2
        while l.size<2: # while l deosn't have 2 points
            lastLine = np.array(img[rows - i, :])
            l = np.where(lastLine > 0)
            l = l[0]
            i = i+1

        sz = l.shape
        p1 = np.array([l[0], rows]).astype(np.float)
        p2 = np.array([l[sz[0]-1], rows]).astype(np.float)
        middle = int((p1[0] + p2[0]) / 2)
        col = np.where(img[:, middle] > 0)
        c = np.amax(col)
        p3 = np.array([middle, c]).astype(np.float)

        # method from: https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/

        x12 = p1[0] - p2[0]
        x13 = p1[0] - p3[0]
        y12 = p1[1] - p2[1]
        y13 = p1[1] - p3[1]
        y21 = p2[1] - p1[1]
        y31 = p3[1] - p1[1]
        x31 = p3[0] - p1[0]
        x21 = p2[0] - p1[0]

        sx13 = pow(p1[0], 2) - pow(p3[0], 2)
        sy13 = pow(p1[1], 2) - pow(p3[1], 2)
        sx21 = pow(p2[0], 2) - pow(p1[0], 2)
        sy21 = pow(p2[1], 2) - pow(p1[1], 2)

        f = ((sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) // (2 * (y31 * x12 - y21 * x13)))
        g = ((sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) // (2 * (x31 * y12 - x21 * y13)))
        c = (-pow(p1[0], 2) - pow(p1[1], 2) - 2 * g * p1[0] - 2 * f * p1[1])

        xc = int(-g)
        yc = int(-f)

  
        rows_mat = np.tile(range(cols), (rows, 1))
        cols_mat = np.tile(np.reshape(range(rows), (rows, 1)), cols)
        pix_dist_mat = np.sqrt(np.power(cols_mat - rows, 2) + np.power(rows_mat - cols/2, 2))

        live_area = np.zeros_like(img)
        live_area[np.where(img>0.0)] = 1
        center_row = np.where(live_area[:, int(cols/2)])
        range_min_idx, range_max_idx = np.max(center_row), np.min(center_row)
        pix_min, pix_max = pix_dist_mat[range_min_idx, int(cols/2)], pix_dist_mat[range_max_idx, int(cols/2)]


        rang_res = (max_range - min_range) / (pix_max - pix_min)



        return [xc,  yc, rang_res]






if __name__ == '__main__':
    # init
    rospy.init_node('StonefishFLS')
    node = StonefishFLS()
    rospy.spin()
