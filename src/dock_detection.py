import cv2
import numpy as np
from numpy import unique

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

import heapq





class dock_detection:

    def __init__(self, min_range = 1.0, max_range = 50.0 , origin_x = 400, origin_y =800 , range_resolution= 0.0018):


        self.min_range = min_range
        self.max_range = max_range
        self.origin_x = origin_x
        self.origin_y = origin_y       
        self.range_resolution = range_resolution

        self.last_image = None  # nir
        self.last_centers = None

    def find_targets(self, mask, target_min_size , target_max_size):
        # print('finding targets')
        # return None, None

        # add a black circle to the mask to remove the robot
        # get the origin of the robot from the center of the image
        origin_x = int(mask.shape[1]/2)
        origin_y = int(mask.shape[0])
        # get the radius of the robot
        radius = int(mask.shape[0]/2)
        scale = 0.5
        # draw the circle
        # mask = cv2.circle(mask, (origin_x, origin_y), int(radius * scale), (0,0,0), -1)




        d_min = np.floor( target_min_size / self.range_resolution )
        d_max = np.floor( target_max_size / self.range_resolution )

        box = []
        centers = []
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        #mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        #_ , contours, hierarchy = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        gray_mask = mask

        #_ , contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        # cv2.imshow("labeled_img", mask)
        # cv2.waitKey(1)

        if contours == []:
            return None, None
        
        #cont = cv2.drawContours(inv, [max(contours, key = cv2.contourArea)], -1, 255, thickness=-1)
        cont = cv2.drawContours(mask, contours, -1, 255, thickness=-1)
        
        
        # mask &= cont



        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)    



        ## Agglomerative clustering
        # img = mask

        # n = 0
        # while n < 3:
        #     img = cv2.pyrDown(img)
        #     n = n + 1

        # rows, cols = img.shape

        # feature_image = np.reshape(img, [-1, 1])

        # # Agglomerative clustering
        # model = AgglomerativeClustering(n_clusters=4)
        # model.fit(feature_image)
        # labels = model.labels_

        # indices = np.dstack(np.indices(img.shape[:2]))
        # xycolors = np.concatenate((img[..., np.newaxis], indices), axis=-1)
        # feature_image2 = np.reshape(xycolors, [-1, 3])
        # model.fit(feature_image2)
        # labels2 = model.labels_

        # output_img = np.reshape(labels2, [rows, cols])
        # output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min()) * 255
        # output_img = output_img.astype(np.uint8)

        # # Find the centers of the clusters
        # centers = []
        # unique_labels = unique(labels2)
        # labels2_reshaped = np.reshape(labels2, [rows, cols])
        # for label in unique_labels:
        #     if label != -1:
        #         center = np.mean(indices[labels2_reshaped == label], axis=0)
        #         centers.append(np.int0(center))

        # return [centers, output_img]





        # DBSCAN
        # img = mask

        # n = 0
        # while n < 3:
        #     img = cv2.pyrDown(img)
        #     n = n + 1

        # rows, cols = img.shape

        # feature_image = np.reshape(img, [-1, 1])

        # print('dbscan')
        # db = DBSCAN(eps=5, min_samples=50, metric='euclidean', algorithm='auto')
        # print('dbscan done')
        # db.fit(feature_image)
        # labels = db.labels_

        # indices = np.dstack(np.indices(img.shape[:2]))
        # xycolors = np.concatenate((img[..., np.newaxis], indices), axis=-1)
        # feature_image2 = np.reshape(xycolors, [-1, 3])
        # db.fit(feature_image2)
        # labels2 = db.labels_

        # output_img = np.reshape(labels2, [rows, cols])
        # output_img = output_img.astype(np.uint8)

        # # Find the centers of the clusters
        # centers = []
        # unique_labels = np.unique(labels2)
        # labels2_reshaped = np.reshape(labels2, [rows, cols])
        # for label in unique_labels:
        #     if label != -1:
        #         center = np.mean(indices[labels2_reshaped == label], axis=0)
        #         centers.append(np.int0(center))

        # return [centers, output_img]


        #inv = cv2.fillPoly(inv,pts=contours,color=(255,255,255))

        #print( contours[1])
        #contours = contours[0]
        #contours = contours[0] if len(contours) == 2 else contours[1]

        mask &= cont

        # print('fillpoly')
        cv2.fillPoly(mask, contours, [255,255,255])
        # print('connected components')
        # gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        num_labels, labels = cv2.connectedComponents(gray_mask)
        # num_labels, labels = cv2.connectedComponents(mask)
        # print('cc done')
        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # Converting cvt to BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        # set bg label to black
        labeled_img[label_hue==0] = 0
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY ) 

        # combine clusters
        #print('agglomerative_cluster ', len(contours))
        if len(contours) > 250:
            if self.last_image is None:
                return [[],blank_ch ]
            else:
                return [self.last_centers, self.last_image]

        contours_clustered = self.agglomerative_cluster(contours= contours,threshold_distance= 10)
        # print('agglomerative_cluster done')
        
        # cv2.imshow("labeled_img", labeled_img)
        # cv2.waitKey(1)

        # filter clusters
        #print('filtering clusters')
        for c in contours_clustered:
        #    x,y,w,h = cv2.boundingRect(c)
        #    cv2.rectangle(labeled_img, (x, y), (x + w, y + h), (255), 2)
        #     bounding_boxes = cv2.rectangle(labeled_img,(x,y),(x+w,y+h),(0,255,0),10)

            #rect = (center(x, y), (width, height), angle of rotation) 
            rect = cv2.minAreaRect(c)

            if  all(d_min <= i <= d_max for i in rect[1]):
                box = cv2.boxPoints(rect)

                if rect[1][0] > rect[1][1]:
                    ratio = rect[1][0] / rect[1][1]

                elif rect[1][0] < rect[1][1]:
                    ratio = rect[1][1] / rect[1][0]

                else:
                    ratio = 1

                if ratio < 1.4:

                    box = np.int0(box)
                    #print(np.int0(rect[0]))
                    centers.append(np.int0(rect[0]))
                    cv2.drawContours(labeled_img,[box],0,(255),2)    
        # imgplot = plt.imshow(labeled_img)
        # plt.show()
        # cv2.waitKey(0)
        #print('finding targets is done')
        # print('centers')
        # print(centers)

        self.last_centers = centers
        self.last_image = labeled_img

        return [centers, labeled_img];
        # return [centers, labels2];



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

    # def agglomerative_cluster(self,contours, threshold_distance=20.0):
    #     current_contours = contours
    #     while len(current_contours) > 1:
    #         print('contours:', len(current_contours))
    #         min_distance = None
    #         min_coordinate = None

    #         for x in range(len(current_contours)-1):
    #             for y in range(x+1, len(current_contours)):
    #                 distance = self.calculate_contour_distance(current_contours[x], current_contours[y])
    #                 if min_distance is None:
    #                     min_distance = distance
    #                     min_coordinate = (x, y)
    #                 elif distance < min_distance:
    #                     min_distance = distance
    #                     min_coordinate = (x, y)

    #         if min_distance < threshold_distance:
    #             index1, index2 = min_coordinate
    #             current_contours[index1] = self.merge_contours(current_contours[index1], current_contours[index2])
    #             del current_contours[index2]
    #         else: 
    #             break

    #     return current_contours



    # def agglomerative_cluster(self, contours, threshold_distance=20.0):
    #     current_contours = contours
    #     distances = []

    #     for x in range(len(current_contours) - 1):
    #         for y in range(x + 1, len(current_contours)):
    #             distance = self.calculate_contour_distance(current_contours[x], current_contours[y])
    #             heapq.heappush(distances, (distance, (x, y)))

    #     while len(current_contours) > 1:
    #         print('contours:', len(current_contours))
    #         min_distance, min_coordinate = heapq.heappop(distances)

    #         if min_distance < threshold_distance:
    #             index1, index2 = min_coordinate
    #             if index1 < len(current_contours) and index2 < len(current_contours):
    #                 current_contours[index1] = self.merge_contours(current_contours[index1], current_contours[index2])
    #                 del current_contours[index2]

    #                 for i in range(len(current_contours)):
    #                     if i != index1:
    #                         distance = self.calculate_contour_distance(current_contours[index1], current_contours[i])
    #                         heapq.heappush(distances, (distance, (index1, i)))
    #         else:
    #             break

    #     return current_contours


    def agglomerative_cluster(self, contours, threshold_distance=20.0):
        current_contours = [(i, contour) for i, contour in enumerate(contours)]
        distances = {}

        for x in range(len(current_contours) - 1):
            for y in range(x + 1, len(current_contours)):
                distance = self.calculate_contour_distance(current_contours[x][1], current_contours[y][1])
                distances[(current_contours[x][0], current_contours[y][0])] = distance

        while len(current_contours) > 1:
            #print('contours:', len(current_contours))
            min_coordinate, min_distance = min(distances.items(), key=lambda x: x[1])

            if min_distance < threshold_distance:
                index1, index2 = min_coordinate
                index1_contour = None
                index2_contour = None

                for idx, contour in enumerate(current_contours):
                    if contour[0] == index1:
                        index1_contour = idx
                    elif contour[0] == index2:
                        index2_contour = idx

                    if index1_contour is not None and index2_contour is not None:
                        break

                merged_contour = self.merge_contours(current_contours[index1_contour][1], current_contours[index2_contour][1])
                current_contours[index1_contour] = (index1, merged_contour)
                del current_contours[index2_contour]

                for i, contour in current_contours:
                    if i != index1:
                        distance = self.calculate_contour_distance(merged_contour, contour)
                        distances[(min(index1, i), max(index1, i))] = distance

                distances = {key: value for key, value in distances.items() if index2 not in key}
            else:
                break

        return [contour for _, contour in current_contours]

   #def CSRT_track(self, mask, d_min, d_max):


