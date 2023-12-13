import cv2
import numpy as np
import munkres
import track
from get_objects import get_objects
from help_functions import find_center


class Kalman_filter:

    def __init__(self):
        self.tracks = []                            # list of tracks 
        self.bboxes = np.empty((0, 6))              # list of track bounding boxes [x, y, w, h, id, visible(-1/1)]
        self.unassignedTracks = np.array([])        # array of the id-s of all the unsigned tracks
        self.unassigned_detection = np.array([])
        self.id = 0
        self.procces_num = 0                        # times the whole proccess ocoured.
        self.color_list = np.empty((0, 3))          # list of the colors for every tracked object
        self.noise_objects = 0
        self.noise_size = 0

    def add_next_frame_features(self, img, mask):
        self.noise_objects = 0
        self.noise_size = 0
        objects_list = np.empty((0, 4)) # x, y, w, h
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        object_sizes = []
        for cnt in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[cnt])
            object_sizes.append(w * h)
            if w>10 or h>10:
                objects_list = np.vstack((objects_list, [x, y, w, h]))
            else:
                self.noise_objects += 1
                self.noise_size = ((self.noise_size * self.noise_objects) + (w * h)) / self.noise_objects
        object_sizes_sorted = np.sort(object_sizes)
        
        self.objects_list = objects_list.astype(np.int)  # list of objects detected in this frame
        self.targetRate = np.ones((objects_list.shape[0], 1))
        self.img = img

    def predict(self):
        for t in self.tracks:
            t.kalman.predict()
            predictedState = t.kalman.x
            t.bbox = np.array([predictedState[0][0], predictedState[2][0], t.bbox[2], t.bbox[3]])

    def costToAssignment(self, cost, costOfNonAssignments):
        # this function get nXm matrix of cost, and the maximum cost, and return assignments of tracks to detections
        # inputs: cost - matrix of nXm (num of tracks X num of detections)
        #        costOfNonAssignments- scalar
        # outputs: assignments- LX2 np.array, where L is the number of detections assigned to tracks
        #                      assignments[:][0] the detections, assignments[:][1] the tracks
        #         unassignedTracks - np array of all the tracks that no detection assigned to them
        #         unassignedDetections - np array of all the detections that not assigned to tracks

        assignments = munkres.Munkres().compute(cost)
        unassignedTracks = np.empty((0, 1), dtype=int)
        unassignedDetection = np.empty((0, 1), dtype=int)
        unAssignedIdx = np.empty((0, 1), dtype=int)
        for i in range(len(assignments)):
            track = assignments[i][0]      # track index
            detection = assignments[i][1]  # detection index
            if cost[track][detection] >= costOfNonAssignments:  # and self.proccesNum>5:
                unassignedTracks = np.vstack((unassignedTracks, track))
                unassignedDetection = np.vstack((unassignedDetection, detection))
                # index from assignments
                unAssignedIdx = np.vstack((unAssignedIdx, i))
        numUnassigned = unAssignedIdx.shape
        for i in range(numUnassigned[0]):
            j = unAssignedIdx[numUnassigned[0] - i - 1]
            del assignments[j[0]]
        return assignments, unassignedTracks, unassignedDetection

    def detectionToTracksAssignment(self):

        self.assignments = np.array([])
        self.unassignedTracks = np.array([])

        nTracks = len(self.tracks)
        nDetections = self.objects_list.shape[0]
        self.unassignedDetection = np.array(range(nDetections))

        if nDetections == 0:
            return
        if nTracks == 0:
            return

        cost = [[0] * nDetections for i in range(nTracks)]  # zeros-matrice list
        costOfNonAssignment = 20
        if nDetections > nTracks:
            d = nDetections - nTracks
            for i in range(d):
                cost.append([costOfNonAssignment + 1] * nDetections)
        if nTracks > nDetections:
            d = nTracks - nDetections
            cost = [x + [costOfNonAssignment + 1] * d for x in cost]
        for i in range(nTracks):
            for j in range(nDetections):
                centroid = np.array([self.objects_list[j][0]+(self.objects_list[j][2]/2), self.objects_list[j][1]+(self.objects_list[j][3]/2)])
                cost[i][j] = np.sqrt(((self.tracks[i].bbox[0] - centroid[0]) ** 2) + \
                                     ((self.tracks[i].bbox[1] - centroid[1]) ** 2))  # centroids need to be np array, as well as tracks['kalmanFilter']
        self.assignments, self.unassignedTracks, self.unassignedDetection = self.costToAssignment(cost, costOfNonAssignment)

        # delete the "zero-padding":
        self.unassignedDetection = self.unassignedDetection[self.unassignedDetection < nDetections]
        self.unassignedTracks = self.unassignedTracks[self.unassignedTracks < nTracks]

    def update(self, snr=0.7):
        invisibleForTooLong = 1 if snr>0.7 else 2
        ageThreshold = 0
        lost = []

        for a in self.assignments:
            trackIdx = a[0]
            detectionIdx = a[1]
            bbox = self.objects_list[detectionIdx][:]
            targetRate = self.targetRate[detectionIdx]
            self.tracks[trackIdx].updateTrack(1, bbox, targetRate)

        for u in self.unassignedTracks:
            self.tracks[u].updateTrack(0) # not visible
            if self.tracks[u].age > ageThreshold:
                if self.tracks[u].invisibleCount > invisibleForTooLong:
                    lost.append(u)
            elif self.tracks[u].visibleCount / self.tracks[u].age > 0.6:
                lost.append(u)

        for l in range(len(lost) - 1, -1, -1):
            del self.tracks[lost[l]]
            self.bboxes[l][5] = -1 # not visible

    def createNewTracks(self):
        unasDet = self.unassignedDetection.shape
        if unasDet[0] == 0:
            return

        # create properties-arrays of the unassigned detections only:
        bboxes = self.objects_list[self.unassignedDetection]
        targetRate = self.targetRate[self.unassignedDetection]

        # create the track objects:
        for i in range(unasDet[0]):
            self.tracks.append(track.track(bboxes[i], self.id, targetRate[i]))
            [x, y, w, h] = self.objects_list[i][:]
            self.bboxes = np.vstack((self.bboxes, [x, y, w, h, self.id, 1]))
            self.color_list = np.vstack((self.color_list, self.tracks[i].color))
            self.id += 1

        self.procces_num = self.procces_num + 1

    def showTracks(self):
        for t in self.tracks:
            p1 = (int(t.bbox[0]), int(t.bbox[1]))
            p2 = (int(t.bbox[0]) + int(t.bbox[2]), int(t.bbox[1]) + int(t.bbox[3]))
            cv2.rectangle(self.img, p1, p2, t.color, 2, 1)
        cv2.imshow('tracking', self.img)
        cv2.waitKey(0)

    def track_it(self, img, mask, range_img, index=0, snr=0.7):
        # tracking and updating
        self.add_next_frame_features(img, mask) 
        self.predict()                       
        self.detectionToTracksAssignment()   
        self.update()                        
        self.createNewTracks()

        return self.bboxes.astype(np.int64), self.color_list.tolist()
