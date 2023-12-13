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
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
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

import tf2_ros
import tf2_geometry_msgs  # for converting PointStamped


from help_functions import transform_from_tf #Yevgeni added for tf
from nav_msgs.msg import Odometry  # orientation of the vehicle

#for topic synchronization
import message_filters

#for rviz markers
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


#for detection test
from secondary_detection import secondary_detection

from dynamic_reconfigure.server import Server
from dynamic_reconfigure.client import Client
from fls_detection.cfg import TutorialsConfig

from itertools import ifilter


from dynamic_reconfigure.server import Server
from dynamic_reconfigure.client import Client
from fls_detection.cfg import TutorialsConfig

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

        self.target_min_size = 1.4 # meters
        self.target_max_size = 3.5 # meters
        self.latch_min_size = 0.1
        self.latch_max_size = 1.3

        # for projection and fusion
        self.sonar_vfov      = 20 * np.pi / 180 / 2 
        self.sonar_ang_res   = 0.003141 # 0.18 deg sonar res in rad
        self.pub_projected   = True
        self.camera_range    = 4
        self.local_range_max = 0
        self.local_range_min = 0

        # for denoising by motion
        self.dx = 0
        self.dy = 0
        self.dpsi = 0

        self.x_prev = 0
        self.y_prev = 0
        self.psi_prev = 0



        # Get namespace and config
        # self.ns = rospy.get_namespace()
        self.ns = '/sparus2/'
        self.target_list = np.array([])
        self.projected_image = np.array([])

        #ranges_sub = message_filters.Subscriber("/sparus2/FLS/Img_range", Image)

        # image_sub  = message_filters.Subscriber("/sparus2/FLS/Img_color/bone/mono", Image)
        # odom_sub   = message_filters.Subscriber("/sparus2/navigator/odometry", Odometry) 
        # sonar_info_sub = message_filters.Subscriber("/sparus2/FLS/sonar_info", sonar_info)

        self.odom_global = None
        self.sonar_global = None
        self.info_global = None
        self.camera_info = None

        self.denoiser = image_denoise()


        self.srv = Server(TutorialsConfig, self.param_callback)
        # self.client = Client("obstacle_avoidance") 

        # self.params = self.client.get_configuration()
        # # print('params', params)
        # self.params['target_min_size'] = 0.8
        # self.params['target_max_size'] = 2.6
        # self.range_reset_counter = 0



        # config = self.srv.update_configuration(self.params)


        
        image_subs = rospy.Subscriber(self.ns + "FLS/Img_color/bone/mono", Image, self.save_sonar_cb)

        #image_subs = rospy.Subscriber("/postprocess/drawn_sonar", Image, self.save_sonar_cb)

        odom_subs = rospy.Subscriber(self.ns + "navigator/odometry", Odometry, self.save_odom_cb)
        info_subs = rospy.Subscriber(self.ns + "FLS/sonar_info", sonar_info, self.save_info_cb)
        # camera_info_subs = rospy.Subscriber(self.ns + "camera/camera_info",CameraInfo,self.camera_info_callback)
        camera_info_subs = rospy.Subscriber(self.ns + "camera/camera_info",CameraInfo,self.camera_info_callback)
        #camera_image_subs= rospy.Subscriber(self.ns + "camera/image_gamma/compressed", CompressedImage, self.camera_mask)
        #camera_image_subs= rospy.Subscriber(self.ns + "camera/image_debayered/compressed", CompressedImage, self.camera_mask)
        camera_image_subs= rospy.Subscriber(self.ns + "camera/image_debayered/compressed", CompressedImage, self.camera_mask)

        #camera_image_subs= rospy.Subscriber(self.ns + "camera/image_color/compressed", CompressedImage, self.camera_mask)
        #camera_image_subs= rospy.Subscriber(self.ns + "camera/image_raw/compressed", CompressedImage, self.camera_mask)
      
        # camera_image_subs= rospy.Subscriber(self.ns + "camera/image_mono/compressed", CompressedImage, self.camera_mask)
        
        
        #camera_image_subs= rospy.Subscriber(self.ns + "camera/image_gamma/compressed", CompressedImage, self.camera_mask)
        #ts = message_filters.ApproximateTimeSynchronizer([image_sub, odom_sub, ranges_sub], queue_size=10,slop=0.1)
        #ts = message_filters.ApproximateTimeSynchronizer([sonar_info_sub, image_sub, image_sub], queue_size=10,slop=0.1)

        # camera_info_subs = rospy.Subscriber(self.ns + "camera_info",CameraInfo,self.camera_info_callback)
        # camera_image_subs= rospy.Subscriber(self.ns + "camera/image_color/compressed", CompressedImage, self.camera_mask)

        #camera_info_subs = rospy.Subscriber("/camera/image_raw/camera_info",CameraInfo,self.camera_info_callback)
        #camera_image_subs= rospy.Subscriber("/camera/image_raw/image_color/compressed", CompressedImage, self.camera_mask)



        # ts = message_filters.ApproximateTimeSynchronizer([image_sub, sonar_info_sub, sonar_info_sub], queue_size=20,slop=100, allow_headerless=True)
        # ts.registerCallback(self.process_fls_info)

        self.pub_denoised = rospy.Publisher(self.ns + 'FLS/Img_denoised', Image, queue_size=10)
        self.pub_denoised_compressed = rospy.Publisher(self.ns + 'FLS/Img_denoised/compressed', CompressedImage, queue_size=1)
        self.pub_circles = rospy.Publisher(self.ns + "FLS/Circles", Image, queue_size=10)
        self.pub_circles_img = rospy.Publisher(self.ns + "FLS/Circles_Image", Image, queue_size=10)
        self.pub_2D_cloud = rospy.Publisher(self.ns + "FLS/point_cloud2D", PointCloud2, queue_size=20)
        self.pub_ned_cloud = rospy.Publisher(self.ns + "FLS/ned_point_cloud", PointCloud2, queue_size=20)
        self.pub_target_markers = rospy.Publisher(self.ns + "FLS/target_markers", Marker, queue_size=20)


        self.pub_projected_image = rospy.Publisher(self.ns + "FLS/projected_image", Image, queue_size=10)
        self.pub_camera_mask = rospy.Publisher(self.ns + "camera/object_mask", Image, queue_size=10)
        self.pub_fuzed_image = rospy.Publisher(self.ns + "fusion/fuzed_image", Image, queue_size=10)
        self.fused_points_cloud = rospy.Publisher(self.ns + "fusion/fused_point_cloud", PointCloud2, queue_size=1)


        # Get transforms
        found = False
        self.tf_handler = TransformHandler()
        self.broadcaster = tf.TransformBroadcaster()

        #    try:
        _, fls_xyz, fls_rpy = self.tf_handler.get_transform(self.ns + 'FLS')
        tf_fls = transform_from_tf(fls_xyz, fls_rpy)      
        self.fls_pos = tf_fls[0]
        self.fls_rot = tf_fls[1]

        # Initialize the TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("fls tf loaded")
        print('Camera - Sonar fusion node initialized')


    def param_callback(self, config, level):
        # rospy.loginfo("""Reconfigure Request: {int_param}, {double_param},\ 
        #     {str_param}, {bool_param}, {size}""".format(**config))
        for key in config:
            print("key", key, config[key])

        self.target_min_size = config['target_min_size']
        self.target_max_size = config['target_max_size']
        self.latch_min_size = config['latch_min_size']
        self.latch_max_size = config['latch_max_size']
        self.denoiser.reconfigure_callback(config)
        return config

    def save_odom_cb(self, msg):
        # print("rcv odom")
        self.odom_global = msg


    def save_sonar_cb(self, msg):
        # print("rcv sonar")
        self.sonar_global = msg

    def save_info_cb(self, msg):
        print("rcv info")
        self.info_global = msg

    def camera_info_callback(self,msg):
        #print('got camera info')
        self.camera_info = msg

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
        print('process fls info')

        odom, info, sonar_info = self.odom_global, self.sonar_global, self.info_global

        if None in [odom, info, sonar_info]:
            return

        rpy = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x,
                                                             odom.pose.pose.orientation.y,
                                                             odom.pose.pose.orientation.z,
                                                             odom.pose.pose.orientation.w])

        #self.ned = np.array([[odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]]).T
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

        dx = odom.pose.pose.position.x - self.x_prev
        self.x_prev = odom.pose.pose.position.x

        dy = odom.pose.pose.position.y - self.y_prev
        self.y_prev = odom.pose.pose.position.y

        dpsi = (rpy[2] - self.psi_prev) * 180 / np.pi
        self.psi_prev = rpy[2]

        dx = sonar_info.range_resolution/dx
        dy = sonar_info.range_resolution/dy
        

        #print('denoise')
        mask = self.denoiser.image_denoise_2(mono_img,dx,dy,dpsi,sonar_info.origin_col, sonar_info.origin_row)
        #mask = d.image_denoise_2(mono_img)
        
        # all objects
        #print('gen range')
        ranges , thetas = gen_range_img(sonar_info.fls_start_range, sonar_info.fls_stop_range, sonar_info.origin_col, sonar_info.origin_row, sonar_info.range_resolution ,mask)

     
        #print('RthetaToCloud')
        self.pub_2D_cloud.publish(self.RthetaToCloud(ranges, thetas))
        #print('RthetaToWorldCloud')
        self.pub_ned_cloud.publish(self.RthetaToWorldCloud(ranges, thetas))
        #print('RthetaToBodyCloud')

        #print('targets_img type', type(targets_img))
        self.pub_denoised.publish(br.cv2_to_imgmsg(mask, encoding="passthrough"))

        #### Create CompressedImage ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', mask)[1]).tostring()
        # Publish new image
        self.pub_denoised_compressed.publish(msg)

        if self.camera_info and self.pub_projected:
             self.pub_projected_img(ranges, thetas)
        
        #print('done')

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

        Y = ranges *   np.sin(thetas) 
        X = ranges *   np.cos(thetas) 
        #x = (X[(X**2 + Y**2)**0.5 > 5.5])
        #y = (Y[(X**2 + Y**2)**0.5 > 5.5])
        z = np.zeros((len(X),), dtype=float) #- self.fls2ned_pos[2]

        x = X.astype(np.float)
        y = Y.astype(np.float)

        points       = np.column_stack([x,y,z])

        points2body  = self.fls_rot.dot(points.T) + self.fls_pos

        points2world = self.rot.dot(points2body) + self.ned

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),]

        header = Header()
        header.frame_id =  "world_ned"
        pc2 = point_cloud2.create_cloud(header, fields, points2world.T)
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



  


 

    def pub_projected_img(self, ranges, thetas):

        h   = self.camera_info.height  
        w   = self.camera_info.width   
        fx  = self.camera_info.K[0] 
        fy  = self.camera_info.K[4] 
        cx  = self.camera_info.K[2] 
        cy  = self.camera_info.K[5] 
        

        self.projected_image = np.zeros((h,w), dtype = np.uint8)

        self.projected_ranges = np.zeros([h,w])

        ranges_fov_range = np.array([])
        thetas_fov_range = np.array([])

        camera_hfov = np.arctan((w/2)/fx)

        #print("ranges", ranges)
        #print("thetas", thetas)
        ranges_fov = (ranges[(abs(thetas) < ( camera_hfov))])
        thetas_fov = (thetas[(abs(thetas) < ( camera_hfov))])

        #print("ranges_fov", ranges_fov)
        #print("thetas_fov", thetas_fov)

        #print("self.camera_range", self.camera_range)



        # ranges_fov_range = (ranges_fov[ranges_fov < self.camera_range])
        # thetas_fov_range = (thetas_fov[ranges_fov < self.camera_range])


        ranges_fov_range = ranges_fov # (ranges_fov[ranges_fov < self.camera_range])
        thetas_fov_range = thetas_fov # (thetas_fov[ranges_fov < self.camera_range])



        #print("ranges_fov_range", ranges_fov_range)
        # print("thetas_fov_range", thetas_fov_range)


  
        max_elevetion    = ranges_fov_range * np.sin(self.sonar_vfov)
        min_elevetion    = -max_elevetion 
        min_elevetion_up = np.clip(max_elevetion, 0,  self.ned[2])

        min_elevetion_up = -1 * min_elevetion_up


        X = ranges_fov_range *   np.sin(thetas_fov_range) 
        Z = ranges_fov_range *   np.cos(thetas_fov_range) 
        Y_max = max_elevetion
        Y_min = min_elevetion_up

        #transform to camera frame (TO DO)

        X_c = X
        Z_c = Z
        Y_c_max = Y_min
        Y_c_min = Y_max 

   

        u = (fx * X_c) / Z_c + cx 
        v_min = (fy * Y_c_max)/ Z_c + cy
        v_max = (fy * Y_c_min)/ Z_c + cy


        u     = u.astype(int)
        v_max = v_max.astype(int)
        v_min = v_min.astype(int)


        # sonar to camera resolution projection

        line_width = fx * (ranges_fov_range * np.sin(self.sonar_ang_res)) / Z_c 
        line_width = np.clip(line_width, 1 , 10)
        line_width  = line_width.astype(int)

        #print("ranges_fov_range", ranges_fov_range, type(ranges_fov_range))
        #print("ranges_fov_range", ranges_fov_range)
        if ranges_fov_range.size !=0:
          
          self.local_range_max = np.amax(ranges_fov_range)
          self.local_range_min = np.amin(ranges_fov_range)
          R_norm  = 255*(ranges_fov_range- self.local_range_min)/(self.local_range_max- self.local_range_min)
          R_norm  = R_norm.astype(int)

        #print(min(Z))

          for i in range(len(u)):
  
            if  0 < u[i] and u[i] < w:
                cv2.line(self.projected_image, tuple([u[i],v_min[i]]), tuple([u[i],v_max[i]]), [R_norm[i],R_norm[i],R_norm[i]], line_width[i]) 

                self.projected_ranges[v_min[i]:v_max[i], u[i]] = Z[i]
                

          #print (self.projected_image.shape)


          self.pub_projected_image.publish(br.cv2_to_imgmsg(self.projected_image, encoding="passthrough"))
          #self.projected_image = projected_image



    def camera_mask(self, camera_image):
        #print("camera cb")
        # return
        # print("camera_image", type(camera_image))

        cv_image = br.compressed_imgmsg_to_cv2(camera_image)

        gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        #gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BayerGR2GRAY)
        # gray_img = cv_image

        # resize image
        # scale_percent = 40 # percent of original size
        # width = int(gray_img.shape[1] * scale_percent / 100)
        # height = int(gray_img.shape[0] * scale_percent / 100)
        # dim = (width, height)
    
        # resized = cv2.resize(gray_img, dim, interpolation = cv2.INTER_AREA)



        gray_blured = cv2.GaussianBlur(gray_img, (5, 5), 0)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(15,15))
        cl1 = clahe.apply(gray_blured)

        thresholded = cv2.adaptiveThreshold(cl1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        thresholded= cv2.medianBlur(thresholded, 9)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        close  = cv2.morphologyEx(np.invert(thresholded), cv2.MORPH_CLOSE,kernel)
        Ex  = cv2.morphologyEx(close, cv2.MORPH_GRADIENT,kernel)

        mask = Ex 

        # cv2.line(mask,(gray_img.shape[1]/4,0),(gray_img.shape[1]/4,gray_img.shape[0]),(255,0,0),10)
        # cv2.line(mask,(gray_img.shape[1]/2,0),(gray_img.shape[1]/2,gray_img.shape[0]),(255,0,0),10)
        # cv2.line(mask,(gray_img.shape[1]*3/4,0),(gray_img.shape[1]*3/4,gray_img.shape[0]),(255,0,0),10)




        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        #contours = contours[0]


        # Applying cv2.connectedComponents()
        num_labels, labels = cv2.connectedComponents(mask)

        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # Converting cvt to BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue==0] = 0

        self.pub_camera_mask.publish(br.cv2_to_imgmsg(mask))
        self.fuse_camera_and_sonar(mask, self.projected_ranges)

    def fuse_camera_and_sonar(self, camera_mask, sonar_mask):
        #print("sonar_mask", type(sonar_mask), sonar_mask)

        if sonar_mask.size == camera_mask.size:


            #fused_image = cv2.bitwise_and(sonar_mask, camera_mask, mask = sonar_mask)
            fused_image = sonar_mask * camera_mask/255

            fused_image_pub = fused_image *255
            fused_image_pub = fused_image_pub.astype(np.uint8)
            # print ("sonar")
            # print (np.max(sonar_mask))
            # print ("fusion")
            # print (np.max(fused_image))

            self.pub_fuzed_image.publish(br.cv2_to_imgmsg(fused_image_pub))


            fx  = self.camera_info.K[0] 
            fy  = self.camera_info.K[4] 
            cx  = fused_image_pub.shape[1]/2  #self.camera_info.K[2] 
            cy  = fused_image_pub.shape[0]/2  #self.camera_info.K[5] 
    

            coordinates = np.where(fused_image > [0.0])
    
            #z_c = (fused_image[coordinates] * (self.local_range_max- self.local_range_min))/255 + self.local_range_min
    
            z_c = fused_image[coordinates]

            x_c = ((coordinates[1] - cx) * z_c)/fx
            y_c = ((coordinates[0] - cy) * z_c)/fy




            # print (max(x_c.astype(np.float)))
            # print (min(x_c.astype(np.float)))
            # print (max(y_c.astype(np.float)))
            # print (min(y_c.astype(np.float)))
            # print (max(z_c.astype(np.float)))
            # print (min(z_c.astype(np.float)))  



            # x = z_c
            # z = y_c
            # y = x_c
    
            x = x_c.astype(np.float)
            y = y_c.astype(np.float)
            z = z_c.astype(np.float)
            points = np.column_stack([x,y,z])
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),]
            header = Header()
            header.frame_id = "sparus2/FLC"
            pc2 = point_cloud2.create_cloud(header, fields, points)
            self.fused_points_cloud.publish(pc2)


    def callback(self, config, level):
        # rospy.loginfo("""Reconfigure Request: {int_param}, {double_param},\ 
        #     {str_param}, {bool_param}, {size}""".format(**config))
        for key in config:
            print("key", key, config[key])

        self.target_min_size = config['target_min_size']
        self.target_max_size = config['target_max_size']
        self.latch_min_size = config['latch_min_size']
        self.latch_max_size = config['latch_max_size']
        return config

if __name__ == '__main__':
    # init
    rospy.init_node('FlsDetection')
    node = FlsDetection()
    #srv = Server(TutorialsConfig, node.callback)
    #client = Client("fls_detection") 

    # params = client.get_configuration()
    # # print('params', params)
    # params['target_min_size'] = 0.8
    # params['target_max_size'] = 2.6
    #config = srv.update_configuration(params)

    r = rospy.Rate(100) # 10hz 
    while not rospy.is_shutdown():
        # pub.publish("hello")
        node.loop()
        r.sleep()

    # rospy.spin()
