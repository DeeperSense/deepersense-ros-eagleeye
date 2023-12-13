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

# Globals
br = CvBridge()
fls_conf = {}
client_dir = os.path.dirname(os.path.realpath(__file__))
yaml = client_dir + os.path.sep + "fls.yaml"
sleep_between_pings = 1
pub = rospy.Publisher('/FLS/Img_mono', Image, queue_size=10)
pub_denoised = rospy.Publisher('/FLS/Img_denoised', Image, queue_size=10)
pub_circles = rospy.Publisher("Circles", Image, queue_size=10)
pub_frontcnt = rospy.Publisher("Front_Contour", Image, queue_size=10)
pub_circles_img = rospy.Publisher("Circles_Image", Image, queue_size=10)
pub_frontcnt_img = rospy.Publisher("Front_Contour_Image", Image, queue_size=10)
pub_point_cloud = rospy.Publisher("Front_Contour_cloud", PointCloud2, queue_size=2)
pub_2D_cloud = rospy.Publisher("/FLS/point_cloud2D", PointCloud2, queue_size=2)


p = process_and_track(track="mixture_particles")
d = image_denoise()
r = get_r_theta()
class navigation_data:
    def __init__(self):
        self.navigation_msg = []

    def nav_listener(self, info):
	self.navigation_msg = info

# global navigation_data:
n_data = navigation_data()


def yaml2dict(yamlPath, objDict):
    if not os.path.exists(yamlPath):
        print("-E- Can't find yaml: " + yamlPath)
        return

    with open(yamlPath, "r") as fp:
        for line in fp.readlines():
            if line.lstrip().startswith("#"): #comment
                continue

            if not line.strip(): #empty
                continue

            key = line.split(":")[0].strip()
            val = line.split(":")[1].strip()
            objDict[key] = val


def process_fls_info(info):

    # sleep(0.2)
    # info is what we get from the FLS-action-server
    # It should contain 3 arrays - sonar image mono, sonar image color and ranges
    br = CvBridge()
    mono_img = cv2.cvtColor(br.imgmsg_to_cv2(info, desired_encoding="passthrough"), cv2.COLOR_RGB2GRAY)
    range_img = min_max_to_range(0.5, 30, mono_img)

    # getting navigation data:
    try:
        rospy.Subscriber('/sparus2/navigator/navigation', NavSts, n_data.nav_listener)
    except:
        print("navigation data did not accepted")

    # predict the state using navigation data
    # if n_data.navigation_msg != []:
        # p.motion_update(n_data.navigation_msg.position, n_data.navigation_msg.orientation, mono_img, range_img)


## yevgeni added this publisher
    mask = d.image_denoise(mono_img , range_img)

    ranges = cv2.bitwise_and(range_img, range_img, mask = mask)
    thetas = r.get_r_theta(mono_img, ranges)

    hight, width = ranges.shape
    ranges = ranges.reshape([hight * width,1])
    thetas = thetas.reshape([hight * width,1])

    #hight, width = ranges.shape
    #hight, width = thetas.shape
    #np.reshape(ranges, (1, hight * width))
    #np.reshape(thetas, (1, hight * width))
    #ranges = ranges.flatten()
    #print(ranges)

    pub_2D_cloud.publish(RthetaToCloud(ranges,thetas))
    #print(range_img)
    #labeled_img = d.denoise_segments(mask)

    #clustered = d.clustering(mask)

    pub_denoised.publish(br.cv2_to_imgmsg(mask, encoding="passthrough"))


    if int(fls_conf["publish_orig_fls_img"]):
        pub.publish(info.data)

    if int(fls_conf["publish_processed_fls_img"]):
        # inputs:
        # img - FLS image
        # range_img - matrix of range to index
        # bool_viz - show or not the objects on image
        [c, f, img_c, img_f, img_b] = p.process_and_track(mono_img, range_img)
        #if c.size and f.size:
            #pub_circles.publish(br.cv2_to_imgmsg(c, encoding="passthrough"))
            #pub_frontcnt.publish(br.cv2_to_imgmsg(f, encoding="passthrough"))
            #pub_point_cloud.publish(fToXYZ(f))
        pub_circles_img.publish(br.cv2_to_imgmsg(img_c, encoding="passthrough"))
        pub_frontcnt_img.publish(br.cv2_to_imgmsg(img_f, encoding="passthrough"))



    # cv2.imshow("fls", mono_img)
    # cv2.waitKey(1000)
    #
    # cv2.imshow("fls", color_img)
    # cv2.waitKey(1000)
    #
    # cv2.imshow("fls", range_img)
    # cv2.waitKey(1000)

#
# def navsts_cb(msg):
#     print("-I- processing navsts")
#     ned = [msg.position.north, msg.position.east, msg.position.down]
#     rpy = [msg.orientation.roll, msg.orientation.pitch, msg.orientation.yaw]



def fToXYZ(f):
    #f[:,1] = f[:,1] - (np.pi)/2
    #print(np.multiply(f[:,1], 180 / 3.14))
    x = np.multiply(f[:,0], np.cos(f[:,1]))
    y = np.multiply(f[:,0], np.sin(f[:,1]))
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
    #print(points[1,:])
    #print("new")

    return pc2

def RthetaToCloud(ranges,thetas):
    #f[:,1] = f[:,1] - (np.pi)/2
    #print(np.multiply(f[:,1], 180 / 3.14))
    X = ranges *  np.sin(thetas)
    Y = ranges *  np.cos(thetas)
    x = (X[(X**2 + Y**2)**0.5 > 5.5])
    y = (Y[(X**2 + Y**2)**0.5 > 5.5])
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
    #print(points[1,:])
    #print("new")

    return pc2




def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/sparus2/FLS/Img_color/bone", Image, process_fls_info)
    #rospy.Subscriber("/FLS/Img_color1/display", Image, process_fls_info)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    print("-I- Starting fls client")
    print("-I- Loading configuration")
    yaml2dict(yaml, fls_conf)
    if int(fls_conf["publish_orig_fls_img"]):
        print("publish_orig_fls_img is true")
    else:
        print("publish_orig_fls_img is false")


    try:

        listener()
    except rospy.ROSInterruptException:
        print("FLS client was interrupted", file=sys.stderr)
