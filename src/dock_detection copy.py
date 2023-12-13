import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import DBSCAN


class dock_detection:

    def __init__(self, min_range = 1.0, max_range = 50.0 , origin_x = 400, origin_y =800 , range_resolution= 0.0018):


        self.min_range = min_range
        self.max_range = max_range
        self.origin_x = origin_x
        self.origin_y = origin_y       
        self.range_resolution = range_resolution

    def find_targets(self, mask, target_min_size , target_max_size):

        d_min = np.floor( target_min_size / self.range_resolution )
        d_max = np.floor( target_max_size / self.range_resolution )

        box = []
        centers = []
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        #mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        #_ , contours, hierarchy = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _ , contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        # cv2.imshow("labeled_img", mask)
        # cv2.waitKey(1)

        if contours == []:
            return None, None
        
        #cont = cv2.drawContours(inv, [max(contours, key = cv2.contourArea)], -1, 255, thickness=-1)
        cont = cv2.drawContours(mask, contours, -1, 255, thickness=-1)
        mask &= cont
        #inv = cv2.fillPoly(inv,pts=contours,color=(255,255,255))

        #print( contours[1])
        #contours = contours[0]
        #contours = contours[0] if len(contours) == 2 else contours[1]
        cv2.fillPoly(mask, contours, [255,255,255])
        num_labels, labels = cv2.connectedComponents(mask)
        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # Converting cvt to BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        # set bg label to black
        labeled_img[label_hue==0] = 0
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY )


        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 120  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 10  # maximum gap in pixels between connectable line segments
        line_image = np.copy(mask) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(mask, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

        # Draw the lines on the  image
        lines_edges = cv2.addWeighted(mask, 0.8, line_image, 1, 0)



        #print labeled_img.shape
        imgplot = plt.imshow(line_image)
        plt.show()
        cv2.waitKey(0)
        # combine clusters agglomerative_cluster method 
        contours_clustered = self.agglomerative_cluster(contours= contours,threshold_distance= 10)

        
        # cv2.imshow("labeled_img", labeled_img)
        # cv2.waitKey(1)

        # filter clusters
        for c in contours_clustered:
        #    x,y,w,h = cv2.boundingRect(c)
        #    cv2.rectangle(labeled_img, (x, y), (x + w, y + h), (255), 2)
        #     bounding_boxes = cv2.rectangle(labeled_img,(x,y),(x+w,y+h),(0,255,0),10)

            #rect = (center(x, y), (width, height), angle of rotation) 
            rect = cv2.minAreaRect(c)

            if  all(d_min <= i <= d_max for i in rect[1]):
                box = cv2.boxPoints(rect)
                print("ok")
                box = np.int0(box)
                #print(np.int0(rect[0]))
                centers.append(np.int0(rect[0]))
                cv2.drawContours(labeled_img,[box],0,(255),2)    
        # imgplot = plt.imshow(labeled_img)
        # plt.show()
        # cv2.waitKey(0)
        return [centers, labeled_img];



    def calculate_contour_distance(self, contour1, contour2): 
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        c_x1 = x1 + w1/2
        c_y1 = y1 + h1/2

        x2, y2, w2, h2 = cv2.boundingRect(contour2)
        c_x2 = x2 + w2/2
        c_y2 = y2 + h2/2

        return max(abs(c_x1 - c_x2) - (w1 + w2)/2, abs(c_y1 - c_y2) - (h1 + h2)/2)

    def merge_contours(self,contour1, contour2):
        return np.concatenate((contour1, contour2), axis=0)

    def agglomerative_cluster(self,contours, threshold_distance=20.0):
        current_contours = contours
        while len(current_contours) > 1:
            min_distance = None
            min_coordinate = None

            for x in range(len(current_contours)-1):
                for y in range(x+1, len(current_contours)):
                    distance = self.calculate_contour_distance(current_contours[x], current_contours[y])
                    if min_distance is None:
                        min_distance = distance
                        min_coordinate = (x, y)
                    elif distance < min_distance:
                        min_distance = distance
                        min_coordinate = (x, y)

            if min_distance < threshold_distance:
                index1, index2 = min_coordinate
                current_contours[index1] = self.merge_contours(current_contours[index1], current_contours[index2])
                del current_contours[index2]
            else: 
                break

        return current_contours


    def find_latch(self, mask,targets_img, target_min_size, target_max_size, ROI):

        d_min = np.floor( target_min_size / self.range_resolution )
        d_max = np.floor( target_max_size / self.range_resolution )

        radiuses = []
        centers = []
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        #mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        #_ , contours, hierarchy = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _ , contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Approximate contours to polygons + get bounding rects and circles
        contours_poly = [None]*len(contours)
        center = [None]*len(contours)
        radius = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            center, radius = cv2.minEnclosingCircle(contours_poly[i])
            if radius >= d_min and radius <= d_max:

                if (np.sqrt(np.power(ROI[0] - int(center[0]), 2) +  np.power(ROI[1] - int(center[1]), 2) ) < 0.5/self.range_resolution):
                    radiuses.append(int(radius))
                    centers.append(np.int0(center))
                    
                    #centers_pix = list(centers)
                    #centers_pix.append((np.int0(centers_pix[0]), np.int0(centers_pix[1])))
                    cv2.circle(targets_img, (int(center[0]), int(center[1])), int(radius), [255,255,255], 2)
        

        return [centers, targets_img];

   #def CSRT_track(self, mask, d_min, d_max):


