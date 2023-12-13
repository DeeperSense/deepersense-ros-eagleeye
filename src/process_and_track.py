import cv2
import numpy as np
import objDetect
#import particle_filter
#import kalman_filter
import mixture_particles_filter
from visualize import show_particles, show_objects
from help_functions import find_center
from get_objects import get_objects
import matplotlib.pyplot as plt
from identification import detect_target2


class process_and_track:

    def __init__(self, track="mixture_particles"):
        if track=="mixture_particles":
            self.tracking_filter = mixture_particles_filter.mixture_particles_filter()
        self.index = 0
        self.first_motion_update = 1

    def motion_update(self, position, orientation, img, range_img):

        self.tracking_filter.motion_update(position, orientation, self.first_motion_update, img, range_img)
        self.first_motion_update = 0
        
        return 0
    
    def process_and_track(self, img, range_img=None, index=1):
        self.index += 1
        
        # pre-processing:
        if self.index == 1:
            self.o = objDetect.ROIfind(img.copy())
        mask = self.o.create_rois_map(img.copy(), range_img)

        # tracking:
        target_box, obstacles_list, colors = self.tracking_filter.track_it(img, mask, range_img, self.index, self.o.snr)

        # get objects convert the list of objects from bounding rectangles to circle (x,y of center and radius) or circles, and front_contours
        circles, circles_for_viz, front_contours, fronts_for_viz, bboxes, colors = get_objects(target_box, obstacles_list, colors, img, mask, range_img, find_center(img))

        # showing results:
        image_c, image_f, image_b = show_objects(mask, circles_for_viz, fronts_for_viz, bboxes, colors, index=self.index)
        
        return circles, front_contours, image_c, image_f, image_b


