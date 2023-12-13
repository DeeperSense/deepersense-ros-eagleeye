import numpy as np
import cv2
import imutils
import objDetect
from secondary_detection import secondary_detection
from help_functions import min_max_to_range , rotate_bound
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
#from skimage.transform import hough_ellipse
#from skimage.feature import canny

s = secondary_detection()
#detector = cv2.SimpleBlobDetector()
def extract_rectangle(img, points, theta):
    #print(points)
    # gets image, points of rectangle, and angle in which the rectangle is rotated, and extract the rectangle.
    x = points[:, 0]
    x[np.where(x<=0)] = 1
    y = points[:, 1]
    y[np.where(y<=0)] = 1
    polly_mask = np.zeros_like(img)
    pts = np.array([[x[i], y[i]] for i in [0,1,3,2]], dtype=np.int32)
    polly_mask = cv2.fillConvexPoly(polly_mask, pts, (255,255,255)) / 255
    mask = polly_mask * img
    # rotate with theta and extract the bounding box
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.ones((2, 2))
    cropped = mask[np.min(y):np.max(y), np.min(x):np.max(x)]
    #rotated = imutils.rotate_bound(cropped, -theta)
    rotated = imutils.rotate_bound(cropped, -theta)
    #find the first and last indices:
    ys, xs = np.where(rotated>0)
    if ys.size == 0 or xs.size == 0:
        return np.zeros((2, 1))
    x_min, y_min = np.min(xs)+1, np.min(ys)+1
    x_max, y_max = np.max(xs)-1, np.max(ys)-1
    cropped_again = rotated[y_min:y_max, x_min:x_max]

    # plt.imshow(cropped_again), plt.show()
    #s.find_target_bulb(img) #yevgeni try
    return cropped_again

# detect thin rectangles (for the buoys on the docking station's sides)
def detect_bouys(mask):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cnt,_ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    lines  = []
    sizes  = []
    angles = []

    for c in cnt:
        min_rect = cv2.minAreaRect(c)
        (x,y), (w,h), angle = min_rect
        if h/w < 1.5: # not thin rectangle
            continue
        box = cv2.boxPoints(min_rect).astype(np.int) # note: box starts from the lowest point and continues clockwise
        rect_mask = cv2.fillConvexPoly(np.zeros_like(mask), box.astype(np.int), 1) * mask
        area = np.sum(rect_mask > 0)
        if min(area/(float(int(w)*int(h))), (float(int(w)*int(h)))/area) > 0.6: # checking if rectangle
            sizes.append(h)
            angles.append(angle)
            if abs(int(np.linalg.norm(box[1]-box[0])) - int(w)) < 2: # if points of the short side is box[0], box[1]
                pts = np.array([(box[0]+box[1])/2, (box[2]+box[3])/2])
            else:
                pts = np.array([(box[3]+box[0])/2, (box[1]+box[2])/2])
            # the highest point first:
            line = np.array([pts[np.argmax(pts[:, 1]), :], pts[np.argmin(pts[:, 1]), :]])
            lines.append(line.astype(np.int))

            # plot for debug: 
            #im = cv2.drawContours(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), [box], 0, (0, 0, 255), 1)
            #im =cv2.line(im, (line[0], line[1]), (line[2], line[3]), (0,255,0))
            #plt.imshow(im)
            #plt.show()
    
    return lines, sizes , angles # lines is a list of 2*2 matrices [[x1, y1], [x2,y2]]

def detect_target(mask):

    target_box = []
    target_rotated_box = []

    lines, sizes, angles = detect_bouys(mask)
    theta_and_i = np.vstack((angles, range(len(angles)))).T
    sorted_theta_and_i = theta_and_i[theta_and_i[:, 0].argsort()]
    
    # find parallel lines:
    parallels = []
    p_angles = []
    rect_masks = []
    for j in range(len(sorted_theta_and_i)-1):
        for k in range(j+1, len(sorted_theta_and_i)):
            if abs(sorted_theta_and_i[j, 0] - sorted_theta_and_i[k, 0]) < 10:
                # indexes of the not sorted angle array:
                i1 = int(sorted_theta_and_i[j, 1])
                i2 = int(sorted_theta_and_i[k, 1])
                line1 = lines[i1]
                line2 = lines[i2]

                # distance between lines:
                d_points = np.linalg.norm(line1[0, :]-line2[0, :])

                # checking if the lines creating rectangle - the sides of the polygon are perpendicular:
                ba = line1[1, :] - line1[0, :]
                bc = line2[0, :] - line2[1, :]

                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)

                if d_points < 100: #and abs(angle - 90) < 15 :
                    
                    p = np.vstack((line1, line2))
                    p_angle = (angles[i1]+angles[i2])/2
                    rect_mask = extract_rectangle(mask, p, p_angle)
                    h, w = rect_mask.shape
                    if w > h and np.sum(rect_mask) > 0.6*w*h:
                        target_rotated_box = np.array([line1[0, :], line1[1, :], line2[1, :], line2[0, :]])
                        box_w = np.max([abs(line1[0, 0]- line2[1, 0]), abs(line1[1, 0]-line2[0, 0])])
                        box_h = np.max([abs(line1[0, 1]- line2[1, 1]), abs(line1[1, 1]-line2[0, 1])])
                        box_x = np.min([line1[0, 0], line1[1, 0], line2[0, 0], line2[1, 0]])
                        box_y = np.min([line1[0, 1], line1[1, 1], line2[0, 1], line2[1, 1]])
                        target_box = [box_x, box_y, box_w, box_h]

    return target_box, target_rotated_box

def bool_target(object_mask, th):
    # getting an object mask. return True if it's docking station and False if not
    
    if object_mask.size == 0:
        return False

    if object_mask.shape[0] > object_mask.shape[1]:
        object_mask = object_mask.T

    # some pre-proccessing to distinguish areas from one amother:
    object_mask = cv2.resize(object_mask, (40, 20))
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    object_mask[np.where(object_mask < th+20)] = 0

    # expected 3 areas
    object_cnt, _ =  cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if len(object_cnt) == 3:
        # plt.imshow(object_mask), plt.show()
        # calculating the x-coordinate of every area:
        centers = np.zeros((3, 2))
        angles = np.zeros((3))
        for j in range(3):
            c1 = object_cnt[j]
            M = cv2.moments(c1)
            centers[j, 0] = int(M['m10']/M['m00']) if M['m00'] else 0
            centers[j, 1] = int(M['m01']/M['m00']) if M['m00'] else 0
            # _, _, angles[j] = cv2.fitEllipse(object_cnt[j])
        # sort centers by x-coordinate
        centers = centers[centers[:, 0].argsort()]
        cx = centers[:, 0]
        cy = centers[:, 1]
        # expected 3 objects. 2 buoys, on the far sides, and the docking station on the middle.
        x_coor_condition = cx[0] < 10 and cx[2] > 30 and cx[1] > 15 and cx[1] < 25
        y_coor_condition = cy[0] <= cy[2] + 5 and cy[0] >= cy[2] - 5
        if x_coor_condition and y_coor_condition:
            return True
        else:
            object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_ERODE, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
            object_mask[np.where(object_mask < th)] = 0
            object_cnt, _ =  cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    #if not 3, than try 4 or 2:
    if len(object_cnt) == 2:
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_ERODE, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        object_mask[np.where(object_mask < th)] = 0
        object_cnt, _ =  cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    if len(object_cnt) == 4 or len(object_cnt) == 3:
        objects_num = len(object_cnt)
        centers = np.zeros((4, 2))
        angles = np.zeros((4))
        for j in range(objects_num):
            c1 = object_cnt[j]
            M = cv2.moments(c1)
            centers[j, 0] = int(M['m10']/M['m00']) if M['m00'] else 0
            centers[j, 1] = int(M['m01']/M['m00']) if M['m00'] else 0
            # _, _, angles[j] = cv2.fitEllipse(object_cnt[j])
        # sort centers by x-coordinate
        centers = centers[centers[:, 0].argsort()]
        cx = centers[:, 0]
        cy = centers[:, 1]
        x_coor_condition = cx[0] < 10 and cx[objects_num-1] > 30
        y_coor_condition = cy[0] <= cy[objects_num-1] + 5 and cy[0] >= cy[objects_num-1] - 5
        if x_coor_condition and y_coor_condition:
            return True


    return False


def detect_target2(mask):

    # detecting the docking station.
    # the station has 3 parts: 2 bouys and the station. 1st stage is to find them all as one object. 2nd to apart them and identify.

    target_box = []
    target_rotated_box = []

    if np.unique(mask).size > 1:
        th = np.unique(mask)[1]

    # dilating the mask to find the docking station as one object
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilating = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    ret, th_dilating = cv2.threshold(dilating, 1, 255,0)
    # plt.imshow(dilating), plt.show()
    cnt, _ = cv2.findContours(th_dilating, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    for i in range(len(cnt)):
        c = cnt[i]
        if len(c)<5:
            continue
        
        # cnt_mask is the mask of the contours from the dilated image put on the original image
        cnt_mask = cv2.drawContours(np.zeros_like(mask), cnt, i, (255,255,255), thickness=-1) / 255
        cnt_mask = cnt_mask * mask

        min_rect = cv2.minAreaRect(c)
        (x1,y1), (w,h), angle = min_rect
        box = cv2.boxPoints(min_rect).astype(np.int)
        object_mask = extract_rectangle(cnt_mask, np.array([box[0], box[1], box[3], box[2]]), angle+90)
        if object_mask.shape[1]>4:
            # open the shape, to see if it has 3 areas
            if bool_target(object_mask, th):
                target_box = cv2.boundingRect(c)
                target_rotated_box = box 

    return target_box, target_rotated_box
