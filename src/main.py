import cv2
import os
import numpy as np
from process_and_track import process_and_track
from help_functions import min_max_to_range, images2video
import matplotlib.pyplot as plt

index = 0
rng = True
name = "doublePlate"
flag_read_range = os.path.isfile("data/range_images/" + name+ ".jpg")

p = process_and_track(track="mixture_particles")
images = []

for i in range(0,100):
    index += 1
    # reading the frame
    image_name = "data/images/" + name + "/img ({}).jpg".format(str(i))
    if os.path.isfile(image_name):
        img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_RGB2GRAY)
    else:
        continue

    '''
    # read or create range_img
    if flag_read_range==1:
        range_img = cv2.cvtColor(cv2.imread("data/range_images/" + name+ ".jpg"), cv2.COLOR_RGB2GRAY)
        range_img = min_max_to_range(0.5, range_img[0, int(range_img.shape[1]/2)], img)
        flag_read_range = 2
    elif flag_read_range==0:
        range_img = min_max_to_range(0.5, 20, img)
        flag_read_range = 2
    '''
    range_img = min_max_to_range(0.05, 8, img)
    [c, f, img_c, img_f, img_b] = p.process_and_track(img, range_img, i)
    
    #plt.imshow(img_f) , plt.show()

images2video(name+".avi", "output_images")
