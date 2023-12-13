#!/usr/bin/env python

import rospy
from teledyne_m900_fls_old.msg import ListOfArrays
import numpy as np
import visualize

def callback(msg):
	d = msg.data
	heights = msg.heights
	rospy.loginfo("heights: {}".format(heights))
	width = msg.width
	rospy.loginfo("width: {}".format(width))
	num_of_lists = msg.num_of_lists
	rospy.loginfo("num: {}".format(num_of_lists))
	data_list = list([])
	index = 0
	for i in range(num_of_lists):
		array_h = np.array([])
		for j in range(heights[i]):
			array_w = np.array([])
			for l in range(width):
				array_w = np.append(array_w, d[index])
				index += 1
			array_h = np.append(array_h, array_w, axis = 0)
		data_list.append(array_h)
	rospy.loginfo("circles: {}".format(data_list))

def listener():
	rospy.init_node('listener', anonymous=True)

	rospy.Subscriber("circles", ListOfArrays, callback)
	# rospy.Subscriber("front_contours", ListOfArrays, callback)
	rospy.spin()

if __name__ == '__main__':
	listener()
