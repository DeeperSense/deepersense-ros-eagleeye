
import numpy as np
from filterpy.kalman import KalmanFilter


class track:
    def __init__(self, bbox, idCount, targetRate=0):
        self.centroid = np.array([bbox[0]+(bbox[2]/2), bbox[1]+(bbox[3]/2)])
        self.state = np.array([[bbox[0]+(bbox[2]/2) ,0, bbox[1]+(bbox[3]/2), 0]])
        self.bbox = bbox
        self.kalman = self.createKalmanTracker()
        self.age = 1
        self.visibleCount = 1
        self.invisibleCount = 0
        self.targetRate = targetRate
        self.id = idCount
        self.color = (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))

    def createKalmanTracker(self):
        dt = 0.03

        kf = KalmanFilter(dim_x=4, dim_z=2)
        state = np.array(self.state)
        kf.x = np.reshape(state, (4, 1))
        kf.F = np.array([[1, dt, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, dt],
                         [0, 0, 0, 1]])  # transition matrix

        kf.H = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0]])  # measurement function

        return kf

    def updateTrack(self, visible, bbox=0, targetRate=0):
        if visible:
            self.centroid = np.array([bbox[0]+(bbox[2]/2), bbox[1]+(bbox[3]/2)])
            self.visibleCount += 1
            self.invisibleCount = 0
            self.targetRate = targetRate
            self.bbox = bbox
            latest_state = self.state.shape[0]
            vx = self.centroid[0] - self.state[latest_state - 1][0]
            vy = self.centroid[1] - self.state[latest_state - 1][2]
            new_state = [self.centroid[0], vx, self.centroid[1], vy]
            self.state = np.vstack((self.state, new_state))
            self.kalman.update(self.centroid)
        else:
            self.invisibleCount += 1
            self.age += 1
