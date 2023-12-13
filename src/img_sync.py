import rospy
import message_filters
from sensor_msgs.msg import Image
from teledyne_m900_fls.msg import combined_image

# Globals
pub = rospy.Publisher('/FLS/Combined', combined_image, queue_size=10)

def callback(fls_sub, flc_sub):
    msg = teledyne_m900_fls.msg.combined_image()
    msg.fls_img = fls_sub
    msg.flc_img = flc_sub
    global pub; pub.publish(msg)


fls_sub = message_filters.Subscriber('/FLS/Img_mono', Image)
flc_sub = message_filters.Subscriber('/camera/image_raw', Image)

ts = message_filters.TimeSynchronizer([fls_sub, flc_sub], 10)
ts.registerCallback(callback)
rospy.spin()