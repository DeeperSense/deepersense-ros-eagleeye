#! /usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import math

class ImageListener:
    def __init__(self, topics):
        self.bridge = CvBridge()
        self.image_dict = {}
        self.received_images = {topic: False for topic in topics} # image received flags
        self.subs = [rospy.Subscriber(topic, Image, self.image_callback, callback_args=topic) for topic in topics]

    def image_callback(self, data, topic):
        try:
            encoding = data.encoding
            cv_image = self.bridge.imgmsg_to_cv2(data, encoding)
        except CvBridgeError as e:
            print(e)
            return
        self.image_dict[topic] = cv_image
        self.received_images[topic] = True

    def all_images_received(self):
        return all(self.received_images.values())

def stack_images(images):
    """
    Takes a list of images and stacks them into a grid that is as close
    to square as possible. I.e. for 6 images, it gives a 2x3 grid.
    """

    # compute integer square root for older Python versions
    grid_size = int(math.sqrt(len(images)))

    while len(images) % grid_size != 0:
        grid_size -= 1

    # resize all images to match the size of the first image in the list
    resized_images = [cv2.resize(img, (images[0].shape[1], images[0].shape[0])) for img in images]

    # make rows from the images
    rows = [np.hstack(resized_images[i:i+grid_size]) for i in range(0, len(images), grid_size)]
    
    # stack the rows to make final image
    img_stack = np.vstack(rows)

    return img_stack

def main():
    rospy.init_node('multi_image_view', anonymous=True)
    topics = [
            '/image_debug0', '/image_debug1', '/image_debug2', 
            '/image_debug3', '/image_debug4', '/image_debug5',
            '/image_debug6', '/image_debug7',
            '/sparus2/FLS/Img_denoised'
              ]

    image_listener = ImageListener(topics)
    pub = rospy.Publisher('combined_image', Image, queue_size=10)
    pub_compressed = rospy.Publisher('combined_image_compressed', CompressedImage, queue_size=10)

    rate = rospy.Rate(10) # adjust to your requirement
    while not rospy.is_shutdown():
        if image_listener.all_images_received():
            images = [image_listener.image_dict[topic] for topic in topics]
            combined_img = stack_images(images)
            # resize
            combined_img = cv2.resize(combined_img, (0,0), fx=0.2, fy=0.2)
            if False:
                cv2.namedWindow('Image Grid', cv2.WINDOW_NORMAL)
                # cv2.namedWindow('Image Grid', cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow('Image Grid', combined_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # publish combined image
            try:
                # pub.publish(image_listener.bridge.cv2_to_imgmsg(combined_img, "bgr8"))
                pub.publish(image_listener.bridge.cv2_to_imgmsg(combined_img, "mono8"))
            except CvBridgeError as e:
                print(e)
            
            try:
                #### Create CompressedImage ####
                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"
                ok, buf = cv2.imencode('.jpg', combined_img)
                if ok:  
                    msg.data = np.array(buf).tostring()
                else:
                    rospy.logwarn("cv2.imencode() has returned an error during jpeg compression.")
                    return
                # Publish new image
                pub_compressed.publish(msg)
            except CvBridgeError as e:
                print(e)
        
        else:
            print('Waiting for all images to be received.')
            topics_to_request = [topic for topic in topics if not image_listener.received_images[topic]]
            print('Requesting images for topics: {}'.format(topics_to_request))
        rate.sleep()

    cv2.destroyAllWindows()

if __name__== '__main__':
    main()