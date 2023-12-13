#! /usr/bin/env python

from __future__ import print_function
import sys
import os
from time import sleep
from tokenize import Double
import rospy
import numpy as np
import actionlib
import cv2
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from cola2_msgs.msg import NavSts
from teledyne_m900_fls.msg import sonar_info
from teledyne_m900_fls.msg import ListOfArrays
from teledyne_m900_fls.msg import FLSAction, FLSGoal, FLSResult, FLSFeedback
from process_and_track import process_and_track #Vered's algo
from image_denoise import image_denoise  #Yevgeni denoising
from dock_detection import dock_detection# find segments
from target_tracking import target_tracking# track the target

from get_r_theta import get_r_theta #Yevgeni find range and bearing
from help_functions import min_max_to_range, get_range, gen_range_img
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


#for detection test
from secondary_detection import secondary_detection


# Globals
br = CvBridge()

client_dir = os.path.dirname(os.path.realpath(__file__))
yaml = client_dir + os.path.sep + "fls.yaml"
sleep_between_pings = 1

p = process_and_track(track="mixture_particles")
d = image_denoise()
r = get_r_theta()
#h = help_functions()
s = secondary_detection()
#detect = dock_detection()

#track = target_tracking()


markerArray = MarkerArray()


class FlsDetection(object):

    def __init__(self):
        """Constructor that gets config, publishers and subscribers."""

        self.target_min_size = 0.2 # meters
        self.target_max_size = 3.5 # meters
        self.latch_min_size = 0.1
        self.latch_max_size = 0.3

        # Get namespace and config
        self.ns = rospy.get_namespace()
        self.target_list = np.array([])

        #ranges_sub = message_filters.Subscriber("/sparus2/FLS/Img_range", Image)

        # image_sub  = message_filters.Subscriber("/sparus2/FLS/Img_color/bone/mono", Image)
        # odom_sub   = message_filters.Subscriber("/sparus2/navigator/odometry", Odometry) 
        # sonar_info_sub = message_filters.Subscriber("/sparus2/FLS/sonar_info", sonar_info)

        self.odom_global = None
        self.sonar_global = None
        self.info_global = None



        image_subs = rospy.Subscriber("/sparus2/FLS/Img_color/bone/mono", Image, self.save_sonar_cb)
        odom_subs = rospy.Subscriber("/sparus2/navigator/odometry", Odometry, self.save_odom_cb)
        info_subs = rospy.Subscriber("/sparus2/FLS/sonar_info", sonar_info, self.save_info_cb)



        #ts = message_filters.ApproximateTimeSynchronizer([image_sub, odom_sub, ranges_sub], queue_size=10,slop=0.1)
        #ts = message_filters.ApproximateTimeSynchronizer([sonar_info_sub, image_sub, image_sub], queue_size=10,slop=0.1)



        # ts = message_filters.ApproximateTimeSynchronizer([image_sub, sonar_info_sub, sonar_info_sub], queue_size=20,slop=100, allow_headerless=True)
        # ts.registerCallback(self.process_fls_info)

        self.pub_denoised = rospy.Publisher(self.ns + 'FLS/Img_denoised', Image, queue_size=10)
        self.pub_circles = rospy.Publisher(self.ns + "FLS/Circles", Image, queue_size=10)
        self.pub_circles_img = rospy.Publisher(self.ns + "FLS/Circles_Image", Image, queue_size=10)
        self.pub_2D_cloud = rospy.Publisher(self.ns + "FLS/point_cloud2D", PointCloud2, queue_size=20)
        self.pub_ned_cloud = rospy.Publisher(self.ns + "FLS/ned_point_cloud", PointCloud2, queue_size=20)
        self.pub_target_markers = rospy.Publisher(self.ns + "FLS/target_markers", Marker, queue_size=20)


        # Get transforms
        found = False
        self.tf_handler = TransformHandler()

        #    try:
        _, fls_xyz, fls_rpy = self.tf_handler.get_transform(self.ns + 'FLS')
        tf_fls = transform_from_tf(fls_xyz, fls_rpy)      
        self.fls_pos = tf_fls[0]
        self.fls_rot = tf_fls[1]

        rospy.loginfo("fls tf loaded")
        print('FLS detection node initialized')

    def save_odom_cb(self, msg):
        # print("rcv odom")
        self.odom_global = msg


    def save_sonar_cb(self, msg):
        # print("rcv sonar")
        self.sonar_global = msg

    def save_info_cb(self, msg):
        # print("rcv info")
        self.info_global = msg

    def loop(self):
        # print("loop")
        if self.odom_global and self.sonar_global and self.info_global:
            # print("success")
            self.process_fls_info()
            self.odom_global = None
            self.sonar_global = None
            self.info_global = None


        # rospy.spi



    # def process_fls_info(self, info,odom, sonar_info):
    def process_fls_info(self):
        print("cbaba")

        odom, info, sonar_info = self.odom_global, self.sonar_global, self.info_global

        rpy = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x,
                                                             odom.pose.pose.orientation.y,
                                                             odom.pose.pose.orientation.z,
                                                             odom.pose.pose.orientation.w])

        self.ned = np.array([[odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]]).T
        # Transform from sensor
        self.rot = tf.transformations.euler_matrix(*rpy)[:3, :3]
        #_, fls_xyz, fls_rpy = self.tf_handler.get_transform(self.ns + 'FLS')
        #print(self.fls_rot)
        targets_xy = []
        latch_xy = []
        latch_pix = []
        # It should contain 3 arrays - sonar image mono, sonar image color and ranges
        mono_img = br.imgmsg_to_cv2(info, desired_encoding='passthrough')
        mask = d.image_denoise_2(mono_img)
        ranges , thetas = gen_range_img(sonar_info.fls_start_range, sonar_info.fls_stop_range, sonar_info.origin_col, sonar_info.origin_row, sonar_info.range_resolution ,mask)

        detect = dock_detection(sonar_info.fls_start_range, sonar_info.fls_stop_range, sonar_info.origin_col, sonar_info.origin_row, sonar_info.range_resolution)
        
        targets_pix, targets_img = detect.find_targets(mask, self.target_min_size, self.target_max_size)
        targets_xy = get_range(sonar_info.origin_col, sonar_info.origin_row, sonar_info.range_resolution, targets_pix)
        selected_targets = self.target_tracking(targets_xy)
        if selected_targets is not None:

            world2body = self.rot.T.dot(selected_targets[:,0:3].T) - self.ned
            body2sonar = self.fls_rot.T.dot(world2body) - self.fls_pos

            sonar2img = np.int0(np.array([body2sonar[:][0]/sonar_info.range_resolution, -body2sonar[:][1]/sonar_info.range_resolution])) #- np.array([sonar_info.origin_col, sonar_info.origin_row])
            sonar2img = (sonar2img.T +  np.array([sonar_info.origin_col, sonar_info.origin_row]))

            for i in range(len(sonar2img)):
                
                if (sonar2img[i][0] > 0) and (sonar2img[i][0] < sonar_info.width )and (sonar2img[i][1] > 0) and (sonar2img[i][1] < sonar_info.height):

                    latch_pix, targets_img =  detect.find_latch(mask, targets_img, self.latch_min_size, self.latch_max_size, sonar2img[i])
                    #print(latch_pix)
                    latch_xy = get_range(sonar_info.origin_col, sonar_info.origin_row, sonar_info.range_resolution, latch_pix)
                    selected_targets = self.target_tracking(latch_xy)

        self.targetMarkers(selected_targets)
        self.pub_2D_cloud.publish(self.RthetaToCloud(ranges, thetas))
        self.pub_ned_cloud.publish(self.RthetaToWorldCloud(ranges, thetas))    
        self.pub_denoised.publish(br.cv2_to_imgmsg(targets_img, encoding="passthrough"))

    def RthetaToCloud(self, ranges, thetas):
        Y = ranges *  np.sin(thetas)
        X = ranges *  np.cos(thetas)
        x = (X[(X**2 + Y**2)**0.5 > 1.5])
        y = (Y[(X**2 + Y**2)**0.5 > 1.5])
        z = np.zeros((len(x),), dtype=float)

        x = x.astype(np.float)
        y = y.astype(np.float)


        points = np.column_stack([x,y,z])
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),]

        header = Header()
        header.frame_id = "sparus2/FLS"
        pc2 = point_cloud2.create_cloud(header, fields, points)
        return pc2

    def RthetaToWorldCloud(self, ranges, thetas):

        Y= ranges *   np.sin(thetas) 
        X = ranges *  np.cos(thetas) 
        #x = (X[(X**2 + Y**2)**0.5 > 5.5])
        #y = (Y[(X**2 + Y**2)**0.5 > 5.5])
        z = np.zeros((len(X),), dtype=float) #- self.fls2ned_pos[2]

        x = X.astype(np.float)
        y = Y.astype(np.float)

        points = np.column_stack([x,y,z])

        points2body = self.fls_rot.dot(points.T) + self.fls_pos

        points2world = self.rot.dot(points2body) + self.ned
  
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),]

        header = Header()
        header.frame_id = "world"
        pc2 = point_cloud2.create_cloud(header, fields, points2body.T)
        return pc2



    def fls2world(self, msg):
        """Get odometry from dynamics to know where the vehicle is."""
        odom = msg

        rpy = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x,
                                                             msg.pose.pose.orientation.y,
                                                             msg.pose.pose.orientation.z,
                                                             msg.pose.pose.orientation.w])

        self.ned = np.array([[odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]]).T
        # Transform from sensor
        self.rot = tf.transformations.euler_matrix(*rpy)[:3, :3]



    def target_tracking(self, targets):

        if targets is not None:
        #if len(targets): 

            targets = np.array(targets)
            selected_targets = np.array([])
            points2body  = self.fls_rot.dot(targets.T) + self.fls_pos
            points2world = self.rot.dot(points2body) + self.ned
            numOfT = len(self.target_list)
            #points2world.T  = np.insert(points2world.T, 3, np.zeros(len(points2world)), axis=1)
            #print(points2world)

            if not (len(self.target_list)):
                self.target_list = points2world.T
                self.target_list  = np.insert(self.target_list, 3, np.zeros(len(self.target_list)), axis=1)

            for i in range(len(points2world.T)):

                d = np.empty(numOfT,dtype=float)  

                for j in range(numOfT): 

                    d[j]= (pow(pow(points2world.T[i][0] - self.target_list[j][0],2) + pow(points2world.T[i][1] - self.target_list[j][1],2),0.5))
                
                if len(d):
                    if np.amin(d) > 2.0: 
                     points2world_with_score = np.append(points2world.T[i], 0)
                     #self.target_list = np.append(self.target_list, [points2world.T[i]], axis=0) 
                     self.target_list = np.append(self.target_list, [points2world_with_score], axis=0)

                    else:
                     min_j = np.where(d == np.amin(d))
                     #self.target_list[min_j] = points2world.T[i]
                     points2world_with_score = np.append(points2world.T[i], (self.target_list[min_j][0][3]+1))
                     self.target_list[min_j] = points2world_with_score

                    sorted_targets = self.target_list[np.argsort(self.target_list[::-1][:, 3])]
  
            indices = np.where(np.array(zip(*self.target_list)[3]) > [25])
            selected_targets = self.target_list[indices[0]]

            return selected_targets


    def targetMarkers(self, obstacles_list):

        if obstacles_list is not None: 

            marker = Marker()
            

            for i in range(len(obstacles_list)):


                marker = Marker()
                marker.header.frame_id = "world"
                marker.type = marker.SPHERE
                #marker.action = marker.DELETE
                marker.action = marker.ADD
                marker.id = i
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = obstacles_list[i][0]
                marker.pose.position.y = obstacles_list[i][1]
                marker.pose.position.z = obstacles_list[i][2]
                self.pub_target_markers.publish(marker)
        
                #for m in markerArray.markers:
                #    m.id = id
                #    id += 1


                #markerArray.markers.append(marker)
                #print('marker.pose.position')
                #print(marker.pose.position)
  
            #self.pub_target_markers.publish(markerArray)
            #return markerArray




if __name__ == '__main__':
    # init
    rospy.init_node('FlsDetection')
    node = FlsDetection()

    r = rospy.Rate(100) # 10hz 
    while not rospy.is_shutdown():
        # pub.publish("hello")
        node.loop()
        r.sleep()

    # rospy.spin()

