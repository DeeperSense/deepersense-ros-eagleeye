import numpy as np
from help_functions import find_center
from scipy.ndimage import binary_fill_holes
import cv2
import os
import math

def pixels_for_ranges(circles_pixels, img, rng_img):

    circles_list = np.zeros_like(circles_pixels)
    
    if rng_img[0, 0] > 0:
        center = find_center(img)
        circles_pixels[np.where(circles_pixels[:,0]>= rng_img.shape[1]), 0] = rng_img.shape[1] - 1
        circles_pixels[np.where(circles_pixels[:,1]>= rng_img.shape[0]), 1] = rng_img.shape[0] - 1
        r = rng_img[circles_pixels[:, 1].astype(int), circles_pixels[:, 0].astype(int)]
        rr_end = circles_pixels[:, 0].astype(int) + circles_pixels[:, 2].astype(int)
        rr_end[np.where(rr_end>=rng_img.shape[1])] = rng_img.shape[1] - 1
        rr = rng_img[circles_pixels[:, 1].astype(int), rr_end]
        theta = np.arctan2(circles_pixels[:, 0] - center[0], circles_pixels[:, 1] - center[1])
    
        circles_list[:, 2] = r - rr
        circles_list[:, 0] = r * np.sin(theta)
        circles_list[:, 1] = r * np.cos(theta)
        #print(r)

    return circles_list

def get_circles(img, mask, bbox, rng_img, radius=30.0):

    if max(bbox[2], bbox[3]) <= 3 * radius:
        circles_list_for_viz = np.array([[bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, 0.5 * max(bbox[2], bbox[3])]])
        circles_list = pixels_for_ranges(circles_list_for_viz, img, rng_img)
        return circles_list, circles_list_for_viz
    
    image = mask[bbox[1]:bbox[1]+bbox[3]+1, bbox[0]:bbox[0]+bbox[2]+1]
    circles_list = np.empty((0,3))
    circles_list_for_viz = np.empty((0,3))
    _, image_th = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    image_th = binary_fill_holes(image_th)
    contours, h = cv2.findContours(image_th.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if len(contours)==0:
        return circles_list, circles_list_for_viz
    c_length = [len(i) for i in contours]
    if np.max(c_length) >= 5:
        (xc, yc), (ma, MA), angle = cv2.fitEllipse(contours[np.argmax(c_length)])

        # small object (MA<2*radius)
        if MA <= 1.5 * radius:
            circles_list_for_viz = np.array([[bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, 0.5 * MA]])
            circles_list = pixels_for_ranges(circles_list_for_viz, img, rng_img)
            return circles_list, circles_list_for_viz
        #else: continue to big objects
        
    else:
        return circles_list_for_viz, circles_list
    
    # big objects
    ys, xs = image.shape
    x, y = np.meshgrid(np.arange(xs), np.arange(ys), sparse=True)
    for i in range(int(-0.5 * ys / radius), int(xs / radius)+1):
        for j in range(int(ys / radius)+1):
            x0 = 2 * radius * (i + 0.5 * j)
            y0 = 2 * radius * np.sqrt(3) / 2 * j

            r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

            indicator = r < radius
            img_indicator = image > 100

            if np.any(image[indicator] != 0):
                mask1 = np.zeros((ys, xs))
                mask1[indicator & img_indicator] = 1
                mask1 = binary_fill_holes(mask1)
                contours, h = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                if not contours:
                    return circles_list_for_viz, circles_list
                
                c_length = [len(c) for c in contours]
                cnt = contours[np.argmax(c_length)]
                if cnt.shape[0] > 5:
                    (x1, y1), (ma, MA), _ = cv2.fitEllipse(cnt)
                    if MA > 5 and MA < 1.5*radius:
                        circles_list = np.vstack([circles_list, [x1 + bbox[0], y1 + bbox[1], int(MA / 2)]])

    # check if there are circles which contained by another one:
    contained = []
    for c1 in circles_list:
        for ind in range(circles_list.shape[0]):
            c2 = circles_list[ind, :]
            if (c1 == c2).all() or c1[2] < c2[2]:
                continue
            distance = np.sqrt(((c1[0] - c2[0]) ** 2) + ((c1[1] - c2[1]) ** 2))
            if distance < c1[2]:
                contained.append(ind)
    contained = np.array(contained)
    if contained.size > 0:
        circles_list = np.delete(circles_list, contained, 0)

    circles_list_for_viz = circles_list
    circles_list = pixels_for_ranges(circles_list_for_viz, img, rng_img)

    return circles_list, circles_list_for_viz


def get_front_contour(img, bbox, rng_img, center):
    xs, ys = int(bbox[0]), int(bbox[1])
    ws, hs = int(bbox[2]), int(bbox[3])
    tiny_gray = img[ys:ys+hs, xs:xs+ws]
    tiny_mask_bw = np.zeros((hs, ws))
    tiny_mask_bw[tiny_gray > 90] = 255
    tiny_mask_bw = binary_fill_holes(tiny_mask_bw)
    mask_bw = np.zeros_like(img)
    mask_bw[ys:ys+hs, xs:xs+ws] = tiny_mask_bw
    contours, _ = cv2.findContours(mask_bw.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    contours = np.array(contours)
    
    front_contours = []
    front_contours_viz = []
    
    for cnt in contours:
        cnt_shape = cnt.shape
        if cnt_shape[0] < 10:
            continue
        cnt = cnt[:, 0]
        rotate_cnt = np.empty_like(cnt, dtype=float)

        rotate_cnt[:, 0] = np.sqrt(((cnt[:, 0]-center[0])**2) + ((cnt[:, 1]-center[1])**2))  # r
        rotate_cnt[:, 1] = np.arctan2(cnt[:, 1] - center[1], cnt[:, 0] - center[0]) +  np.pi/2  # theta
        ind = 0
        s = rotate_cnt.shape
        r = np.interp(np.linspace(0, s[0], s[0]*5), np.linspace(0, s[0], s[0]), rotate_cnt[:, 0])
        theta = np.interp(np.linspace(0, s[0], s[0]*5), np.linspace(0, s[0], s[0]), rotate_cnt[:, 1])
        theta = np.round_(theta, 3) # resolution: 0.001 radian

        front_cnt = []
        theta_unique = np.unique(theta)
        r_min = 0
        for t in theta_unique:
            loc = np.where(theta == t)
            loc = loc[0]
            if loc.size < 2:
                continue
            r_min1 = np.min(r[loc])
            if r_min1 - r_min < 5 or r_min == 0:
                r_min = r_min1
                front_cnt.append([r_min, t])

        front_cnt1 = np.array(front_cnt)
        n_front = len(front_cnt)
        if n_front == 0:
            continue
        x = np.abs(front_cnt1[:, 0] * np.cos(front_cnt1[:, 1]-np.pi) + center[0])
        x[np.where(x>img.shape[1]-1)] = img.shape[1]-1
        y = np.abs(front_cnt1[:, 0] * np.sin(front_cnt1[:, 1]-np.pi) + center[1])        
        y[np.where(y>img.shape[0]-1)] = img.shape[0]-1
        front_cnt = np.array(front_cnt)
        front_cnt[:, 0] = x
        front_cnt[:, 1] = y

        front_contours_viz.append(front_cnt.copy())
        if rng_img[0, 0] > 0:
            front_cnt[:, 0] = rng_img[y.astype(int), x.astype(int)]  # r
            front_cnt[:, 1] = front_cnt1[:, 1]                       # theta

        front_contours.append(front_cnt)

    # whats happened here?
    list_of_lengthes = np.array([len(i) for i in front_contours])
    if list_of_lengthes.shape[0] > 0:
        mm = np.argmax(list_of_lengthes)
        f = front_contours[mm]
        if rng_img[0, 0] > 0:
            f_for_viz = front_contours_viz[mm]
        else:
            f_for_viz = []
    else:
        f = []
        f_for_viz = []

    return np.array(f), f_for_viz


def get_objects(target, obstacles_list, colors_list, img, mask, rng_img, center):
    # get the list of bounding rectangles of list and the target, if the target apear in the frame, and convert it to the wanted output
    # INPUTS: 
    # target - bounding rect of the target, if apears in the frame
    # obstacle_list - bounding rects of other objects, if apears in the frame, with the index and the visibility (x,y,w,h,i,v)
    # color_list - obstacles colors organized by index
    # img, mask (mask is the output of objDetect), rng_img
    # center - the computed position of the sonar in (or out of) the image
    # OUTPUTS:
    # circles and circles_for_viz - the center, radius and index of the bounding circles of the target and obstacles (x,y,r,index). target index - 0
    # front_contour and front_countors_for_viz - for every target, the list of pixels (in range and pixels) of the front contour (x,y,index)
    # colors - colors of the target and object organized by the index, for vizualization

    circles =         np.empty((0,4))
    front_contours =  np.empty((0,3))
    circles_for_viz = []
    fronts_for_viz =  []
    bbox_for_viz = []
    colors = np.empty((0, 3))

    if target != []:
        target_circles, target_circles_for_viz = get_circles(img, mask, target, rng_img)
        if target_circles.any():
            c_row = np.zeros((len(target_circles), 1))
            target_circles = np.hstack((target_circles, c_row))
        else:
            target_circles = np.empty((0,4))
        circles = np.vstack((circles, target_circles))
        circles_for_viz.append(target_circles_for_viz)

        front_contour, f_for_viz = get_front_contour(img, target, rng_img, center)
        if front_contour.any():
            f_index_row = np.zeros((len(front_contour), 1))
            front_contour = np.hstack((front_contour, f_index_row))
        else:
            front_contour = np.empty((0,3))
        front_contours = np.vstack((front_contours, front_contour))
        fronts_for_viz.append(f_for_viz)

        bbox_for_viz.append(target)

        colors = np.vstack((colors, np.array([255, 0, 0])))

    for t in obstacles_list:
        if t[5] >= 3 : # detected in this frame
            one_obj_circles, one_obj_circles_for_viz = get_circles(img, mask, t, rng_img)
            if one_obj_circles.any():
                c_index_row = np.ones((len(one_obj_circles), 1)) * t[4]
                one_obj_circles = np.hstack((one_obj_circles, c_index_row))
            else:
                one_obj_circles = np.empty((0,4))
            circles = np.vstack((circles, one_obj_circles))
            circles_for_viz.append(one_obj_circles_for_viz)

            front_contour, f_for_viz = get_front_contour(img, t, rng_img, center)
            if front_contour.any():
                f_index_row = np.ones((len(front_contour), 1)) * t[4]
                front_contour = np.hstack((front_contour, f_index_row))
            else:
                front_contour = np.empty((0,3))
            front_contours = np.vstack((front_contours, front_contour))
            fronts_for_viz.append(f_for_viz)

            bbox_for_viz.append(t[:4])

            colors = np.vstack((colors, colors_list[t[4]]))  # the color is in the index-num of color_list
    #print(circles)
    return circles, circles_for_viz, front_contours, fronts_for_viz, bbox_for_viz, colors

