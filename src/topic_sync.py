#! /usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry  # orientation of the vehicle

rospy.init_node('sync_test')

def callback(image, camera_info):
    print('hi')# Solve all of perception here...



image_sub = message_filters.Subscriber('/FLS/Img_color1/display', Image)
info_sub = message_filters.Subscriber('/sparus2/dynamics/odometry', Odometry)

ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub], queue_size=10,slop=0.1)
ts.registerCallback(callback)
rospy.spin()