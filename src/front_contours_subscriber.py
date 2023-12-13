import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

def callback(data):
    bridge = CvBridge()
    front_contours = bridge.imgmsg_to_cv2(data, "bgr8")
    rospy.loginfo("front_contours: {}".format(front_contours))
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('fc_listener', anonymous=True)

    rospy.Subscriber("Front_Contours", 	Image, callback)


    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
