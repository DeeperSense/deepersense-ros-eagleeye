import numpy as np
from numpy.random import uniform
import cv2
from get_objects import get_objects
import mixture_track
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visualize import show_circles_and_bboxes
#from skimage.segmentation import flood_fill
from help_functions import find_center
from identification import detect_target, detect_target2
from motion_model import motion_model

class mixture_particles_filter:

    def __init__(self):
        self.tracks_list=[]
        self.idx = 0
        self.color_list = np.zeros((1, 3))
        self.target_track = []


    def birth_model(self, index, img, mask):
        # create new tracks from the edges of the image

        bm_mask = np.zeros_like(mask)
        bm_mask[np.where(mask>0)] = 1
        for t in self.tracks_list:
            [x, y, l_x, l_y] = t.state
            l = np.where(bm_mask[y:y+l_y, x:x+l_x])
            while l[0].size>0:
                cv2.floodFill(bm_mask, None, (l[1][0]+x, l[0][0]+y), 0)
                l = np.where(bm_mask[y:y+l_y, x:x+l_x])
        mask = mask * bm_mask
        min_obj_size = 50
        
        self.create_tracks(img, mask, index, min_size_obj=min_obj_size)


    def create_tracks(self, img, mask, index, min_size_obj=30):
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
	self.max_n_particles = 50
        if index==1:
            self.max_n_particles = 50 #int(1e4/len(contours))
        for cnt in range(len(contours)):
            x, y, w1, h1 = cv2.boundingRect(contours[cnt])
            x, y = np.max([[x-5, y-5], [0,0]], axis=0)
            x_end, y_end = np.min([[x+w1+10, y+h1+10], [mask.shape[1], mask.shape[0]]], axis=0)
            w, h = [x_end-x, y_end-y]
            if w1*h1 > min_size_obj and min(float(h1)/img.shape[0], float(w1)/img.shape[1]) < 0.5:
                self.idx += 1
                self.color_list = np.vstack((self.color_list, [int(uniform(0, 255)), int(uniform(0, 255)), int(uniform(0, 255))]))
                self.tracks_list.append(mixture_track.mixture_track(x, y, w, h, self.idx, mask, self.max_n_particles))


    def death_model(self, objects_list):
        # two reasons to kill tracks:
        if not objects_list:
            return []
        # 1. overlap objects:
        over_idxes = []
        obj = np.array(objects_list)
        sorted_list = obj[np.argsort(obj[:, 0])] # sorted by x coordinate
        for idx in range(len(sorted_list)-1):
            for j in range(1, len(sorted_list)-idx):
                xl1, yu1, l_x1, l_y1 = sorted_list[idx, :4]
                xl2, yu2, l_x2, l_y2 = sorted_list[idx+j, :4]
                xr1, yd1 = xl1+l_x1, yu1+l_y1
                xr2, yd2 = xl2+l_x2, yu2+l_y2
                left_upp = [max(xl1, xl2), max(yu1, yu2)]
                rigt_dwn = [min(xr1, xr2), min(yd1, yd2)]
                overlap_area = max(0, rigt_dwn[0]-left_upp[0]) * max(0, rigt_dwn[1]-left_upp[1])
                # if the overlap area is more then 0.4 of one of the objects, the object with bigger age:
                #if overlap_area>0.0:
                if np.max([float(overlap_area)/float(l_x1*l_y1), float(overlap_area)/float(l_x2*l_y2)]) > 0.25:
                    if sorted_list[idx, 5] == sorted_list[idx+j, 5]: # if the objects are at the same age
                        delete_i = np.argmin([l_x1*l_y1, l_x2*l_y2])
                    else:
                        delete_i = np.argmin([sorted_list[idx,5], sorted_list[idx+j, 5]])
                    over_idxes.append(sorted_list[idx+(j*delete_i), 4])

        if over_idxes:
            overlaps = []
            overlaps = np.array([np.where(obj[:,4] == i)[0] for i in over_idxes])[:,0] # convert tracks indexes to tracks_list indexes
            #self.tracks_list = [i for j, i in enumerate(self.tracks_list) if j not in overlaps] # delete overlaps
            for i in overlaps: # almost delete overlaps
                self.tracks_list[i].visibility = min(-2, self.tracks_list[i].visibility-1)
                
        # 2. not see object anymore:
        lost = []
        for idx, t in enumerate(self.tracks_list):
            if t.visibility <= -3:
                lost.append(idx)
        
        if lost:
            self.tracks_list = [i for j, i in enumerate(self.tracks_list) if j not in lost]

        # update objects_list:
        objects_list = []
        [objects_list.append(np.hstack((t.state, t.idx, t.visibility))) for t in self.tracks_list]
        
        return objects_list
    

    def show_particles_and_estimations(self, img):
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for t in self.tracks_list:
            [x, y, lx, ly] = t.state
            rect = patches.Rectangle((x, y), lx, ly, linewidth=5, edgecolor=self.color_list[t.idx],facecolor='none')
            ax.add_patch(rect)
            for p in t.particles:
                rect = patches.Rectangle((p[0], p[1]), p[2], p[3], linewidth=1, edgecolor=self.color_list[t.idx],facecolor='none')
                ax.add_patch(rect)

        plt.show()
    

    def target_tracking(self, mask, th, objects_list, index):

        target_box = []
        obstacles_list = objects_list

        # find the target        
        found_target_box, target_rotated_box = detect_target2(mask)

        # if target found:
        if found_target_box != []:
            # new target_track
            if self.target_track == []:
                x, y, l_x, l_y = found_target_box
                self.target_track = mixture_track.mixture_track(x, y, l_x, l_y, 0, mask, 100)
            # or tracking an existed one:
            else:
                self.target_track.state = found_target_box
                self.target_track.update_properties()
                self.target_track.visibility = max(1, self.target_track.visibility+1)
        
        # if not found target:
        else:
            # if existed target_track, and the targt not visible at this image, continue the tracking
            if self.target_track != []:
                self.target_track.predict()
                self.target_track.update(mask, th)
                self.target_track.estimate()
                # self.target_track.visibility -= 1
        
        # after trying to find the target manualy and through the tracking, if found, delete track or tracks overlapped with it:
        if self.target_track != [] and self.target_track.visibility > 1:
            target_box = self.target_track.state
            overlapped_obstacles = []
            xl_t, yu_t, l_x_t, l_y_t = target_box
            xr_t, yd_t = xl_t+l_x_t, yu_t+l_y_t
            for i in range(len(objects_list)):
                xl_o, yu_o, l_x_o, l_y_o = objects_list[i][0:4]
                xr_o, yd_o = xl_o+l_x_o, yu_o+l_y_o
                left_upp = [max(xl_t, xl_o), max(yu_t, yu_o)]
                rigt_dwn = [min(xr_t, xr_o), min(yd_t, yd_o)]
                overlap_area = max(0, rigt_dwn[0]-left_upp[0]) * max(0, rigt_dwn[1]-left_upp[1])
                if overlap_area>0.0:
                    overlapped_obstacles.append(i)
            obstacles_list = [i for j, i in enumerate(objects_list) if j not in overlapped_obstacles]

        return obstacles_list, target_box
    
        
    def motion_update(self, position, orientation, first_update_bool, img, range_img):
        
        position = np.array([position.north, position.east, position.depth])
        orientation = np.array([orientation.roll, orientation.pitch, orientation.yaw])
        
        if first_update_bool or img==[]:
            self.position = position
            self.orientation = orientation
            return
    
        center = find_center(img)
        d_pos = position - self.position
        d_ort = orientation - self.orientation
        
        self.position = position
        self.orientation = orientation
        
        for t in self.tracks_list:
            pred_x, pred_y, pred_z = motion_model(t.state, center, range_img, d_pos, d_ort)
            t.state[0] = pred_x
            t.state[1] = pred_y
    

    def track_it(self, img, mask, range_img, index, snr=0.7):

        rang = np.unique(mask)
        if rang.shape[0] > 1:
            r = 255.0 - rang[1]
            th_weight = (rang[1] + 0.3*r) / 255.0
        else:
            th_weight = 0.5
        
        for t in self.tracks_list:
            t.predict()
            t.update(mask, th_weight)
            t.estimate()

        self.birth_model(index, img, mask)

        objects_list = []
        [objects_list.append(np.hstack((t.state, t.idx, t.visibility))) for t in self.tracks_list]

        objects_list = self.death_model(objects_list)

        obstacles_list, target_box = self.target_tracking(mask, th_weight, objects_list, index)

        # for debuging:
        #show_circles_and_bboxes(img, circles_for_viz, colors, objects_list)
        #self.show_particles_and_estimations(img)
        
        return target_box, obstacles_list, self.color_list
