#!/usr/bin/env python

import rospy

from dynamic_reconfigure.server import Server
from fls_detection.cfg import TutorialsConfig

def callback(config, level):
    # rospy.loginfo("""Reconfigure Request: {int_param1}, {double_param},\ 
    #       {str_param}, {bool_param}, {size}""".format(**config))

    for key in config:
        rospy.loginfo("Key: " + key + " Value: " + str(config[key]))
    return config

if __name__ == "__main__":
    rospy.init_node("dynamic_tutorials", anonymous = False)

    srv = Server(TutorialsConfig, callback)
    rospy.spin()