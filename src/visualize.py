# coding: utf-8
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc


def show_circles(img, circles, color_list, save, index, name):
    for i in range(len(circles)):
        c = circles[i]
        if len(c.shape) > 1:
            for one_c in c:
                img = cv2.circle(img, (int(one_c[0]), int(one_c[1])), int(one_c[2]), color=color_list[i])
    #cv2.imwrite("output_images/circles_frame_{:03d}.png".format(index), img)

    return img


def show_front_contour(img, front_contours, color_list, save, index, name):
    
    for i in range(len(front_contours)):
        f = np.array(front_contours[i])
        if f.any():
            f = f.astype(np.int)
            img[f[:, 1], f[:, 0]] = color_list[i]
    #cv2.imwrite("output_images/fronts_frame_{:03d}.png".format(index), img)
    return img

def show_bboxes(img, bboxes, color_list, index):
    for i in range(len(bboxes)):
        b = bboxes[i]
        img = cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), color=color_list[i])
    cv2.imwrite("output_images/rects_{:03d}.png".format(index), img)
    return img

def show_objects(img, circles, front_contours, bboxes, color_list, name='name', save_c=False, save_f = False, index=0):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    image_c = show_circles(img.copy(), circles, color_list, save_c, index, name)
    image_f = show_front_contour(img.copy(), front_contours, color_list, save_f, index, name)
    image_b = show_bboxes(img, bboxes, color_list, index)

    return image_c, image_f, image_b

def show_circles_and_bboxes(img, circles, color_list, objects_list, index=0):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    obj_ind = 0
    for ind in range(len(circles)):
        while objects_list[obj_ind][5] < 0:
            obj_ind += 1
        x, y, w, h,_,_ = objects_list[obj_ind]
        rect = plt.Rectangle((x, y), w, h, edgecolor=color_list[ind], facecolor='none')
        ax.add_patch(rect)
        c = circles[ind]
        if len(c.shape) > 1:
            for one_c in c:
                circle = plt.Circle((one_c[0], one_c[1]), one_c[2], fill=False, color=color_list[ind])
                ax.add_patch(circle)
        obj_ind += 1
    plt.show() 

def images_to_video():
    im_fold = \
        'C:/Users/vered/Documents/IOLR/‏‏Find-and-Tracking-FLS-objects-master -debug/results/exp_24_05_8/particles/'
    video_name = 'particles_video.avi'
    images = [img for img in os.listdir(im_fold) if img.endswith(".png")]
    frame = cv2.imread(im_fold + images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(im_fold, image)))

    cv2.destroyAllWindows()
    video.release()


def show_particles(particles, img):
    if len(img.shape) == 2:
        row, col = img.shape
        img1 = np.zeros((row, col, 3))
        img1[:, :, 0] = img
        img1[:, :, 1] = img
        img1[:, :, 2] = img
        img = img1
    red = [0, 0, 255]
    for particle in particles:
        img[particle[0],particle[1]] = red
    plt.imshow(img)




