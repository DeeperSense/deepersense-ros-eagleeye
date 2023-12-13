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

from process_and_track import process_and_track #Vered's algo
from image_denoise import image_denoise  #Yevgeni denoising
from dock_detection import dock_detection# find segments
from target_tracking import target_tracking# track the target

from get_r_theta import get_r_theta #Yevgeni find range and bearing
from help_functions import min_max_to_range, get_range, gen_range_img_oculus
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, PointStamped
#for transformations
from cola2.utils.ned import NED
from cola2_ros.diagnostic_helper import DiagnosticHelper
from cola2_ros.transform_handler import TransformHandler
import tf
from help_functions import transform_from_tf #Yevgeni added for tf
from nav_msgs.msg import Odometry  # orientation of the vehicle

import tf2_ros
import tf2_geometry_msgs  # for converting PointStamped


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

from std_msgs.msg import Int64MultiArray

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


        # for projection and fusion
        self.sonar_vfov      = 20 * np.pi / 180 / 2 
        self.sonar_ang_res   = 0.003141 # 0.18 deg sonar res in rad
        self.pub_projected   = True
        self.camera_range    = 30
        self.local_range_max = 0
        self.local_range_min = 0
        self.scale_factor = 0.2
        self.sonar_inclination = 15 * np.pi /180


        self.fls_start_range = 0
        self.fls_stop_range = 40
        self.fls_angle      = 130 * np.pi / 180

        self.timestamp = 0

        # for denoising by motion
        self.dx = 0
        self.dy = 0
        self.dpsi = 0

        self.x_prev = 0
        self.y_prev = 0
        self.psi_prev = 0


        self.fabian_x = []
        self.fabian_y = []
        self.fabian_r = []
        self.fabian_t = []


        # Get namespace and config
        self.ns = rospy.get_namespace()
        self.target_list = np.array([])
        self.projected_image = np.array([])


        self.projected_ranges = None
        #self.projected_image = None
        #ranges_sub = message_filters.Subscriber("/sparus2/FLS/Img_range", Image)

        # image_sub  = message_filters.Subscriber("/sparus2/FLS/Img_color/bone/mono", Image)
        # odom_sub   = message_filters.Subscriber("/sparus2/navigator/odometry", Odometry) 
        # sonar_info_sub = message_filters.Subscriber("/sparus2/FLS/sonar_info", sonar_info)

        # self.odom_global = None
        # self.sonar_global = None
        # self.info_global = None
        # self.camera_info = None

        
        self.denoiser = image_denoise()


        #self.srv = Server(TutorialsConfig, self.param_callback)
        #self.client = Client("obstacle_avoidance") 

        #self.params = self.client.get_configuration()
        # print('params', params)
        #self.params['target_min_size'] = 0.8
        #self.params['target_max_size'] = 2.6


        #config = self.srv.update_configuration(self.params)

        self.camera_width = 0
        self.camera_height = 0 

        self.resized_img = None

        image_subs = message_filters.Subscriber("/oculus/drawn_sonar_rect", Image)
        xy_sonar_subs = message_filters.Subscriber("/oculus/drawn_sonar", Image)
        odom_subs = message_filters.Subscriber("/sparus2/navigator/odometry", Odometry)
        odom_subs = message_filters.Subscriber("/sparus2/navigator/navigation", NavSts)
        camera_info_subs = message_filters.Subscriber("/camera/camera_info",CameraInfo)
        #camera_image_subs= message_filters.Subscriber("/processed_image", Image)
        camera_image_subs= message_filters.Subscriber("/camera/image_raw", Image)

        # Synchronize the topics
        # ts = message_filters.TimeSynchronizer([image_subs, odom_subs, camera_info_subs, camera_image_subs], 10)
        # ts = message_filters.TimeSynchronizer([odom_subs, camera_info_subs], 10)
        ts = message_filters.ApproximateTimeSynchronizer([image_subs, image_subs, camera_info_subs, camera_image_subs,xy_sonar_subs], queue_size=1, slop=0.1)
        ts.registerCallback(self.general_callback)


        # Publishers
        self.pub_denoised = rospy.Publisher(self.ns + 'FLS/Img_denoised', Image, queue_size=10)
        self.pub_denoised_compressed = rospy.Publisher(self.ns + 'FLS/Img_denoised/compressed', CompressedImage, queue_size=1)

        self.pub_2D_cloud = rospy.Publisher(self.ns + "FLS/point_cloud2D", PointCloud2, queue_size=1)

        self.pub_projected_image = rospy.Publisher(self.ns + "FLS/projected_image", Image, queue_size=10)
        self.pub_camera_mask = rospy.Publisher(self.ns + "camera/object_mask/compressed", CompressedImage, queue_size=10)
        self.pub_fuzed_image = rospy.Publisher(self.ns + "fusion/fuzed_image", Image, queue_size=10)
        self.fused_points_cloud = rospy.Publisher(self.ns + "fusion/fused_point_cloud", PointCloud2, queue_size=1)
        #self.fused_points_cloud = rospy.Publisher("voxel_grid/output", PointCloud2, queue_size=1)
        self.pub_bright = rospy.Publisher(self.ns + 'FLS/FLS_bright', Image, queue_size=10)

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


    def camera_info_callback(self,msg):
        #print('got camera info')
        self.camera_info = msg

    def loop(self):

        if self.odom_global and self.sonar_global :
            #print("success")
            self.process_fls_info()
            self.odom_global = None
            self.sonar_global = None
            self.info_global = None
            self.sonar_rows = None
            self.sonar_cols = None

        # rospy.spi


    def general_callback(self, odom_msg, sonar_image_msg, camera_info, camera_image, xy_sonar):
    # def process_fls_info(self, info,odom, sonar_info):
    # def process_fls_info(self):
        #print('process fls info')

        self.timestamp = sonar_image_msg.header.stamp


        # FLC
        cv_xy_sonar = br.imgmsg_to_cv2(xy_sonar, desired_encoding="bgr8")

        cv_xy_sonar = cv2.flip(cv_xy_sonar, 1) 

        # Define the brightness factor (adjust as needed)
        brightness_factor = 4
        contrast_factor = 5
        # Create an array filled with zeros of the same size as the image
        brightness_matrix = np.zeros_like(cv_xy_sonar)

        # Add the brightness factor to the image
        #cv_xy_sonar = cv2.addWeighted(cv_xy_sonar, 1, brightness_matrix, 0, brightness_factor)

        # Apply contrast adjustment formula
        cv_xy_sonar = contrast_factor  * (cv_xy_sonar) 

        # Clip pixel values to the valid range [0, 255]
        cv_xy_sonar = np.clip(cv_xy_sonar, 0, 255)

        # Convert the pixel values back to uint8
        cv_xy_sonar = cv_xy_sonar.astype(np.uint8)

        self.pub_bright.publish(br.cv2_to_imgmsg(cv_xy_sonar , encoding="bgr8"))

        cv_image = self.camera_mask(camera_image)
        # print(cv_image)

        # orig_width = 3384 
        # orig_hight = 2710
        # self.camera_width = cv_image.shape[1]
        # self.camera_height = cv_image.shape[0]  

        odom = odom_msg
        # sonar_image_msg

        if None in [odom]:
            return

        # rpy = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x,
        #                                                 odom.pose.pose.orientation.y,
        #                                                 odom.pose.pose.orientation.z,
        #                                                 odom.pose.pose.orientation.w])
        rpy = [0 , 0, 0]
        self.ned = np.array([[0, 0, 0]]).T

        # rpy = [odom.orientation.roll , odom.orientation.pitch, odom.orientation.yaw]
        # self.ned = np.array([[odom.position.north, odom.position.east, odom.position.depth]]).T

        #self.ned = np.array([[odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]]).T
        # Transform from sensor
        #self.rot = tf.transformations.euler_matrix(*rpy)[:3, :3]
        #_, fls_xyz, fls_rpy = self.tf_handler.get_transform(self.ns + 'FLS')
        #print(self.fls_rot)

        # It should contain 3 arrays - sonar image mono, sonar image color and ranges
        #print("ok")
        mono_img = br.imgmsg_to_cv2(sonar_image_msg, desired_encoding='mono8')

        # Create a black border at the bottom of the image 100 pixels high
        
        #mono_img = cv2.copyMakeBorder(mono_img, 0, 100, 0, 0, cv2.BORDER_CONSTANT, value=0)        

        self.sonar_rows,  self.sonar_cols  = mono_img.shape


        dx = self.ned[0] - self.x_prev
        self.x_prev = self.ned[0]

        dy = self.ned[1] - self.y_prev
        self.y_prev = self.ned[1]

        dpsi = (rpy[2] - self.psi_prev) * 180 / np.pi
        self.psi_prev = rpy[2]

        #dx = sonar_info.range_resolution/dx
        #dy = sonar_info.range_resolution/dy
        
        #mono_img = cv2.flip(mono_img, 1)
        #print('denoise')
        mask = self.denoiser.image_denoise_oculus(mono_img,dx,dy,dpsi, self.sonar_cols, self.sonar_rows)
        #mask = d.image_denoise_2(mono_img)
        #print('mask')
        
        # all objects
        #print('gen range')
        #mask = cv2.copyMakeBorder(mask, 0, 100, 0, 0, cv2.BORDER_CONSTANT, value=0)
        ranges , thetas = gen_range_img_oculus( self.fls_start_range, self.fls_stop_range , self.fls_angle , mask)

        ranges = ranges * np.cos(self.sonar_inclination)
        #print(np.max(ranges))
     
        thetas = thetas[ranges > 1.5]
        ranges = ranges[ranges > 1.5]

        #print('RthetaToCloud'
        #### Create CompressedImage ####
        msg = CompressedImage()
        msg.header.stamp = self.timestamp
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', mask)[1]).tostring()
        # Publish new image
        self.pub_denoised_compressed.publish(msg)

        #print(self.camera_info)
        # if self.camera_info and self.pub_projected:
        if sonar_image_msg and self.pub_projected:
             #print('done')
             self.pub_projected_img(ranges, thetas, camera_info)


        self.fuse_camera_and_sonar(camera_info, cv_image, self.projected_ranges, self.timestamp, camera_image)
        self.pub_2D_cloud.publish(self.RthetaToCloud(ranges, thetas))
        self.pub_denoised.publish(br.cv2_to_imgmsg(mask, encoding="passthrough"))

        #print('done')

    def RthetaToCloud(self, ranges, thetas):

        #print('ok')
        Y = ranges *  np.sin(thetas)
        X = ranges *  np.cos(thetas)
        x = (X[(X**2 + Y**2)**0.5 > 2])
        y = (Y[(X**2 + Y**2)**0.5 > 2])
        z = np.zeros((len(x),), dtype=float)

        x = x.astype(np.float)
        y = y.astype(np.float)


        points = np.column_stack([x,y,z])
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),]

        header = Header()
        header.frame_id = "sparus2/FLS"
        header.stamp = self.timestamp

        pc2 = point_cloud2.create_cloud(header, fields, points)
        return pc2



    def pub_projected_img(self, ranges, thetas, camera_info):

        h   =  int(camera_info.height  * self.scale_factor)
        w   =  int(camera_info.width   * self.scale_factor )
        fx  =  int(camera_info.K[0]    * self.scale_factor)
        fy  =  int(camera_info.K[4]    * self.scale_factor)
        cx  =  int(camera_info.K[2]    * self.scale_factor )
        cy  =  int(camera_info.K[5]    * self.scale_factor )
        


        # if self.camera_width > 0 :
        #     w = self.camera_width
        #     h = self.camera_height

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
          if abs(self.local_range_max- self.local_range_min) > 0:
            R_norm  = 255*(ranges_fov_range- self.local_range_min)/abs(self.local_range_max- self.local_range_min)
            R_norm  = R_norm.astype(int)

        #print(min(Z))

            for i in range(len(u)):
    
                if  0 < u[i] and u[i] < w :
                    cv2.line(self.projected_image, tuple([u[i],v_min[i]]), tuple([u[i],v_max[i]]), [R_norm[i],R_norm[i],R_norm[i]], line_width[i]) 

                    self.projected_ranges[v_min[i]:v_max[i], u[i]] = Z[i]

            #print (self.projected_image.shape)

            # camera_image =    br.imgmsg_to_cv2(camera_image)#(camera_image, cv2.COLOR_BGR2BGRA)
            # camera_image = cv2.resize(camera_image, ( int(self.camera_width*self.scale_factor),  int(self.camera_height*self.scale_factor)), interpolation = cv2.INTER_AREA)


            # camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2BGRA)

            #projected_image = cv2.cvtColor(self.projected_image, cv2.COLOR_BGR2BGRA)

            #projected_image[np.all(projected_image == [0, 0, 0, 255], axis=-1)] = [0, 0, 0, 0]


            # Merge the two images
            #merged_image = cv2.addWeighted(projected_image, 1, camera_image, 1, 0)

            #self.pub_projected_image.publish(br.cv2_to_imgmsg(camera_image, encoding="passthrough"))
            self.pub_projected_image.publish(br.cv2_to_imgmsg(self.projected_image, encoding="passthrough"))
            #self.projected_image = projected_image



    # def camera_mask(self, camera_image):
    #     # print("camera cb")
    #     # return
    #     # print("camera_image", type(camera_image))

    #     cv_image = br.imgmsg_to_cv2(camera_image)
    # #     orig_width = 3384
    # #     orig_hight = 2710
       
    #     self.camera_width = cv_image.shape[1]
    #     self.camera_height = cv_image.shape[0]  


    def camera_mask(self, camera_image):
        #start = time.time()

        kernel_size = 5
        
            #print("camera cb")
        cv_image = br.imgmsg_to_cv2(camera_image)

        self.camera_width = cv_image.shape[1]
        self.camera_height = cv_image.shape[0] 
        # print(cv_image.shape)


        # resize the image
        cv_image = cv2.resize(cv_image, ( int(self.camera_width*self.scale_factor),  int(self.camera_height*self.scale_factor)), interpolation = cv2.INTER_AREA)
        self.resized_img = cv_image
        gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BayerGR2GRAY)

        gray_blured = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(kernel_size,kernel_size))
        cl1 = clahe.apply(gray_blured)

        thresholded = cv2.adaptiveThreshold(cl1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        thresholded= cv2.medianBlur(thresholded, kernel_size)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
        close  = cv2.morphologyEx(np.invert(thresholded), cv2.MORPH_CLOSE,kernel)
        Ex  = cv2.morphologyEx(close, cv2.MORPH_GRADIENT,kernel)

        mask = Ex 
        #contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Define the top-left and bottom-right coordinates of the rectangle

        # Define the color of the rectangle (in BGR format)
        color = (0, 0, 0)  # Black color

        top_left =   (0, int(self.camera_height*self.scale_factor)-50)  # (x, y) coordinates of the top-left corner
        bottom_right = (int(self.camera_width*self.scale_factor), int(self.camera_height*self.scale_factor))  # (x, y) coordinates of the bottom-right corner

        # Draw a black rectangle on the image
        mask = cv2.rectangle(mask, top_left, bottom_right, color, thickness=cv2.FILLED)

        top_left =   (int(self.camera_width*self.scale_factor)-30, int(self.camera_height*self.scale_factor)-230)  # (x, y) coordinates of the top-left corner
        bottom_right = (int(self.camera_width*self.scale_factor), int(self.camera_height*self.scale_factor))  # (x, y) coordinates of the bottom-right corner


        # Draw a black rectangle on the image
        mask = cv2.rectangle(mask, top_left, bottom_right, color, thickness=cv2.FILLED)

        #color = (40, 40, 40)  # Black color

        top_left =   (int(self.camera_width*self.scale_factor)-50, int(self.camera_height*self.scale_factor)-350)  # (x, y) coordinates of the top-left corner
        bottom_right = (int(self.camera_width*self.scale_factor), int(self.camera_height*self.scale_factor))  # (x, y) coordinates of the bottom-right corner

        # Draw a black rectangle on the image
        mask = cv2.rectangle(mask, top_left, bottom_right, color, thickness=cv2.FILLED)

        mask_img = br.cv2_to_imgmsg(mask)

        msg = CompressedImage()
        #msg.header.stamp = msg.header.stamp #rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', mask)[1]).tostring()

        self.pub_camera_mask.publish(msg)

        return mask
        #print("Time after msg: " + str(time.time() - start))


        #timestamp = camera_image.header.stamp

        #self.fuse_camera_and_sonar(camera_info, cv_image, self.projected_ranges, timestamp)

    def dual_net_cb(self, msg):
        # print("dual net cb")
        # print("msg", msg.data)

        for i in range(0, len(msg.data), 4):
            print("x: " + str(msg.data[i]) + " y: " + str(msg.data[i+1]) + " r: " + str(msg.data[i+2]) + " t: " + str(msg.data[i+3]))
            x = msg.data[i]
            y = msg.data[i+1]
            r = msg.data[i+2]
            t = msg.data[i+3]

            print('self.sonar_rows', self.sonar_rows)

            if self.sonar_rows == None:
                return

            # range_resolution = (self.fls_stop_range  - self.fls_start_range) / float(self.sonar_rows)
            range_resolution = (self.fls_stop_range  - self.fls_start_range) / (self.sonar_rows)
            theta_resolution = self.fls_angle /  float(self.sonar_cols) 


            ranges = range_resolution * ((self.sonar_rows) - r)
            thetas = theta_resolution * (t - self.sonar_cols / 2)

            fx  = int(self.camera_info.K[0] )
            fy  = int(self.camera_info.K[4] )
            #cx  = int(fused_image_pub.shape[1]/2  #self.camera_info.K[2]   * scale_factor
            #cy  = int(fused_image_pub.shape[0]/2  #self.camera_info.K[5] 
            cx  = int(self.camera_width/2)
            cy =  int(self.camera_height/2)

            #z_c = (fused_image[coordinates] * (self.local_range_max- self.local_range_min))/255 + self.local_range_min
    
            z_c = ranges * np.cos(thetas)
            x_c = ((x - cx) * z_c)/fx
            y_c = ((y - cy) * z_c)/fy


            x_p = x_c.astype(np.float)
            y_p = y_c.astype(np.float)
            z_p = z_c.astype(np.float)


            points = np.column_stack([x_p,y_p,z_p])
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),]
            header = Header()
            header.stamp = self.timestamp
            header.frame_id = "sparus2/base_link"
            pc2 = point_cloud2.create_cloud(header, fields, points)
            # print("pc2", pc2)
            self.fused_points_cloud.publish(pc2)


            # publish

    def fuse_camera_and_sonar(self, camera_info, camera_mask, sonar_mask, time_stamp,camera_img):
        # print("fuse")
        #print("sonar_mask", type(sonar_mask), sonar_mask)

        orig_width = 3384
        #orig_hight = 2710



        width = float(camera_mask.shape[1])
        height = float(camera_mask.shape[0]) 

        #self.scale_factor = 1
        #print(scale_factor)
        if sonar_mask is not None:

            if sonar_mask.size == camera_mask.size:


                #fused_image = cv2.bitwise_and(sonar_mask, camera_mask, mask = sonar_mask)
                fused_image = sonar_mask * camera_mask/255

                fused_image_pub = fused_image *255
                fused_image_pub = fused_image_pub.astype(np.uint8)
                # print ("sonar")
                # print (np.max(sonar_mask))

                # print (np.max(fused_image))



            # camera_image =    br.imgmsg_to_cv2(camera_image)#(camera_image, cv2.COLOR_BGR2BGRA)
            # camera_image = cv2.resize(camera_image, ( int(self.camera_width*self.scale_factor),  int(self.camera_height*self.scale_factor)), interpolation = cv2.INTER_AREA)


            # camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2BGRA)

                fused_image_pub = cv2.cvtColor(fused_image_pub, cv2.COLOR_BGR2BGRA)

                fused_image_pub[np.all(fused_image_pub == [0, 0, 0, 255], axis=-1)] = [0, 0, 0, 0]

                non_black_pixels = (fused_image_pub[:, :, :3] != [0, 0, 0]).any(axis=2) & (fused_image_pub[:, :, 3] != 0)


                fused_image_pub[non_black_pixels] = [0, 255, 0, 255]  # Set to green (BGR with alpha)

                camera_img =    br.imgmsg_to_cv2(camera_img)#(camera_image, cv2.COLOR_BGR2BGRA)
                camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2BGRA)
                camera_img = cv2.resize(camera_img, ( int(self.camera_width*self.scale_factor),  int(self.camera_height*self.scale_factor)), interpolation = cv2.INTER_AREA)

            # Merge the two images
                merged_image = cv2.addWeighted(fused_image_pub, 1, camera_img, 1, 0)

            #self.pub_projected_image.publish(br.cv2_to_imgmsg(camera_image, encoding="passthrough"))


                self.pub_fuzed_image.publish(br.cv2_to_imgmsg(merged_image, encoding="bgra8"))


                fx  = int(camera_info.K[0] * self.scale_factor)
                fy  = int(camera_info.K[4] * self.scale_factor)
                #cx  = int(fused_image_pub.shape[1]/2  #self.camera_info.K[2]   * scale_factor
                #cy  = int(fused_image_pub.shape[0]/2  #self.camera_info.K[5] 
                cx  = int(width/2)
                cy =  int(height/2)
                coordinates = np.where(fused_image > [0.0])
        
                #z_c = (fused_image[coordinates] * (self.local_range_max- self.local_range_min))/255 + self.local_range_min
        
                z_c = fused_image[coordinates]

                x_c = ((coordinates[1] - cx) * z_c)/fx
                y_c = ((coordinates[0] - cy) * z_c)/fy


                x = x_c.astype(np.float)
                y = y_c.astype(np.float)
                z = z_c.astype(np.float)

                points = np.column_stack([x,y,z])
                fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),]
                header = Header()
                header.frame_id = "sparus2/FLC"
                header.stamp = self.timestamp

                pc2 = point_cloud2.create_cloud(header, fields, points)
                self.fused_points_cloud.publish(pc2)
                # try:
                #             # Get the transform from input_frame to output_frame
                #             transform = self.tf_buffer.lookup_transform("world_ned", "sparus2/FLC", rospy.Time(0), rospy.Duration(1.0))

                #             # Transform each point in the cloud
                #             transformed_points = []
                #             for point in points:
                #                 point_stamped = PointStamped()
                #                 point_stamped.header.frame_id = "sparus2/FLC"
                #                 point_stamped.point.x = point[0]
                #                 point_stamped.point.y = point[1]
                #                 point_stamped.point.z = point[2]

                #                 # Use tf2_geometry_msgs to do the transformation
                #                 transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform).point
                #                 transformed_points.append((transformed_point.x, transformed_point.y, transformed_point.z))

                #                 # Create a new PointCloud2 message with the transformed points
                                
                #                 transformed_cloud = point_cloud2.create_cloud_xyz32(header, transformed_points)


                # except tf2_ros.LookupException as e:
                #     rospy.logerr("Transform lookup failed: %s", e)
                # except tf2_ros.ExtrapolationException as e:
                #     rospy.logerr("Transform extrapolation failed: %s", e)
                # except tf2_ros.TransformException as e:
                #     rospy.logerr("Transform failed: %s", e)

                #self.fused_points_cloud.publish(transformed_cloud)


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
    rospy.init_node('FusionNode')
    node = FlsDetection()
    #srv = Server(TutorialsConfig, node.callback)
    #client = Client("fls_detection") 

    #params = client.get_configuration()
    # print('params', params)
    #params['target_min_size'] = 0.8
    #params['target_max_size'] = 2.6



    #config = srv.update_configuration(params)

    # r = rospy.Rate(100) # 10hz 
    # while not rospy.is_shutdown():
    #     # pub.publish("hello")
    #     node.loop()
    #     r.sleep()

    rospy.spin()
