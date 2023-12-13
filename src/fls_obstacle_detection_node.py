#! /usr/bin/env python

from __future__ import print_function
import sys
import os
from time import sleep
import rospy
import numpy as np
import actionlib
import cv2
import struct
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from cola2_msgs.msg import NavSts
from teledyne_m900_fls.msg import ListOfArrays
from teledyne_m900_fls.msg import FLSAction, FLSGoal, FLSResult, FLSFeedback
from process_and_track import process_and_track #Vered's algo
from image_denoise import image_denoise #Yevgeni denoising
from get_r_theta import get_r_theta #Yevgeni find range and bearing
from help_functions import min_max_to_range
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

#for transformations
from cola2.utils.ned import NED
from cola2_ros.diagnostic_helper import DiagnosticHelper
from cola2_ros.transform_handler import TransformHandler
import tf
from help_functions import transform_from_tf #Yevgeni added for tf
from nav_msgs.msg import Odometry  # orientation of the vehicle

#for topic synchronization
import message_filters
from sensor_msgs.msg import Image, CameraInfo

#for rviz markers
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


# Globals
br = CvBridge()

client_dir = os.path.dirname(os.path.realpath(__file__))
yaml = client_dir + os.path.sep + "fls.yaml"
sleep_between_pings = 1

p = process_and_track(track="mixture_particles")
d = image_denoise()
r = get_r_theta()
#h = help_functions()

markerArray = MarkerArray()


class FlsObstacleDetection(object):

    def __init__(self):
        """Constructor that gets config, publishers and subscribers."""
        # Get namespace and config
        self.ns = rospy.get_namespace()

        self.pub_denoised = rospy.Publisher(self.ns + 'FLS/Img_denoised', Image, queue_size=10)
        #self.pub_2D_cloud = rospy.Publisher(self.ns + "FLS/point_cloud2D", PointCloud2, queue_size=20)
        #self.pub_ned_cloud = rospy.Publisher(self.ns + "FLS/ned_point_cloud", PointCloud2, queue_size=20)


        # Get transforms
        found = False
        self.tf_handler = TransformHandler()
        while (not found) and (not rospy.is_shutdown()):
            try:
                _, fls_xyz, fls_rpy = self.tf_handler.get_transform(self.ns + 'FLS')
                tf_fls = transform_from_tf(fls_xyz, fls_rpy)
                self.fls_pos = tf_fls[0]
                self.fls_rot = tf_fls[1]

                rospy.loginfo("fls tf loaded")

                found = True
            except Exception as e:
                rospy.logwarn("cannot find FLS transform")
                rospy.logwarn(e.message)
                rospy.sleep(2.0)
                #exit(1) This should not exit as sometimes not all tfs are available on the first run
        print('FLS detection node initialized')

        #rospy.Subscriber("/FLS/Img_color1/display", Image, self.process_fls_info)
        # Odometry subscriber
        #rospy.Subscriber(self.ns + 'dynamics/odometry', Odometry, self.fls2world, queue_size=1)


        #image_sub = message_filters.Subscriber("/FLS/Img_color1/display", Image)
        #image_sub = message_filters.Subscriber("/sparus2/FLS/Img_color/bone", Image)
        image_sub = rospy.Subscriber("/sparus2/FLS/Img_color/bone", Image, self.process_fls_info)
        #odom_sub  = message_filters.Subscriber(self.ns + 'dynamics/odometry', Odometry)
        #odom_sub  = message_filters.Subscriber("/sparus2/navigator/odometry", Odometry)
        #ts = message_filters.ApproximateTimeSynchronizer([image_sub, odom_sub], queue_size=10,slop=0.5)
        #ts.registerCallback(self.process_fls_info)
        #self.sync(Image, Odometry)



    #def process_fls_info(self, info,odom):
    def process_fls_info(self, info):
        #scale_percent = 50 # percent of original size
        #width = int(info.shape[1] * scale_percent / 100)
        #height = int(info.shape[0] * scale_percent / 100)
        #dim = (width, height)
        # resize image
        #info = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #def process_fls_info(self, info, Odometry):
    # It should contain 3 arrays - sonar image mono, sonar image color and ranges
        mono_img = cv2.cvtColor(br.imgmsg_to_cv2(info, desired_encoding="passthrough"), cv2.COLOR_RGB2GRAY)
        range_img = min_max_to_range(0.5, 20, mono_img)
        mask = d.image_denoise(mono_img , range_img)
   
        self.pub_denoised.publish(br.cv2_to_imgmsg(mask, encoding="passthrough"))



if __name__ == '__main__':
    # init
    rospy.init_node('Fls_Obstacle_Detection')
    node = FlsObstacleDetection()
    rospy.spin()