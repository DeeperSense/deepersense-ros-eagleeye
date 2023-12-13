import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import cv2
#from identification import bool_target
class mixture_track:

    def __init__(self, x, y, l_x, l_y, idx, img, max_n_particles):
        # create a mixture particle tracker, with the following parameters:
        self.N = np.min([max_n_particles, l_x*l_y])                        # number of particles. 600 is maximum particles
        self.state = np.array([x, y, l_x, l_y], dtype=np.int32)            # initialize state
        self.covs = np.array([0.1*l_x, 0.1*l_y, min(0.1*l_x, 5), min(0.1*l_y, 5)])       # state covariances: sigma_x, sigma_y, sigma_lx, sigma_ly
        self.particles = np.zeros((self.N, 4))                             # list of particles for predict
        self.weights = np.zeros((self.N))                                  # list of weights for update
        self.idx = idx                                                     # index number of the tracker
        self.img = img
        self.visibility = 0                                                # for how long the object is visible
        self.age = 1
        self.obj_properties = {}                                           # properties of objects
        self.update_properties()


    def update_properties(self):       
        x, y, l_x, l_y = self.state
        map = self.img[y:y+l_y+1, x:x+l_x+1]
        th = np.unique(self.img)[1] if len(np.unique(self.img))>1 else 0 # the next value above 0 is the threshold
        _, bw_map = cv2.threshold(map, th, 255, 0)
        cntr, _ = cv2.findContours(bw_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        if len(cntr) == 0:
            width, height, angle_factor, angle = [0,0,0,0]
            xc, yc = x + int(l_x/2), y + int(l_y/2)
        else:
            lengths = [len(cnt) for cnt in cntr]
            if np.max(lengths) < 6:
                width, height, angle_factor, angle = [0,0,0,0]
                xc, yc = x + int(l_x/2), y + int(l_y/2)
            else:
                (xc,yc),(MA,ma),angle = cv2.fitEllipse(cntr[np.argmax(lengths)])
                #(x1,y1), (w,h), angle1 = cv2.minAreaRect(cntr[np.argmax(lengths)])
                angle_factor = MA/ma
                if self.age > 1:
                    width = (self.obj_properties["width"] * (min(self.age, 10) -1) + l_x) / min(self.age, 10)
                    height = (self.obj_properties["height"] * (min(self.age, 10) -1) + l_y) / min(self.age, 10)
                    if self.age > 30:
                        debug =1
                else:
                    width, height = l_x, l_y
        
        area = np.sum(map)
        density = float(np.sum(map > th)) / (l_x * l_y)

        self.obj_properties = {
            "map": map,
            "width": width,
            "height": height,
            "center": [xc, yc],
            "angle_factor": angle_factor,
            "angle": angle,
            "area": area,
            "density": density,
            "n_objects": len(cntr)
        }


    def predict(self):
        self.pred_state = self.state
        
        # probability normal distribution, clipped on the edges of the image:
        x_predict = np.clip(np.random.normal(self.pred_state[0], self.covs[0], self.N), 0, self.img.shape[1]-1)
        y_predict = np.clip(np.random.normal(self.pred_state[1], self.covs[1], self.N), 0, self.img.shape[0]-1)
        lx_predict = np.clip(np.random.normal(self.pred_state[2], self.covs[2], self.N), 5, self.img.shape[1] - self.pred_state[0])
        ly_predict = np.clip(np.random.normal(self.pred_state[3], self.covs[3], self.N), 5, self.img.shape[0] - self.pred_state[1])
        self.particles = np.dstack((x_predict, y_predict, lx_predict, ly_predict))
        self.particles = self.particles[0].astype(np.uint16)
        self.particles[np.where(x_predict + lx_predict >= self.img.shape[1]), 2] =\
            self.img.shape[1] - x_predict[np.where(x_predict + lx_predict >= self.img.shape[1])]
        self.particles[np.where(y_predict + ly_predict >= self.img.shape[0]), 3] =\
            self.img.shape[0] - y_predict[np.where(y_predict + ly_predict >= self.img.shape[0])]
        self.weights = np.ones((self.N, 1))


    def update(self, observation_map, th_weight, target_box = []):
        # the update step updating the predicted particles' weights by the observation of the next frame. 
        # the parameters of the particles compared to the parameters of the last object estimation.

        self.img = observation_map
        if np.unique(self.img).size >= 2:
            th = np.unique(self.img)[1]
        else:
            th = 0

        #if target:
            #self.particles = np.vstack((self.particles, target_box))
            #self.N = self.N + 1
        target_found = np.zeros((self.N))
        
        [x, y, l_x, l_y] = [self.particles[:, i] for i in range(4)]
        y_end, x_end = np.min([(self.img.shape*np.ones((x.shape[0], 2))).T, [y+l_y, x+l_x]], axis=0).astype(np.uint16)

        # properties to measure the particles:
        rmse = np.zeros((self.N))
        density = np.zeros((self.N))
        angle_similarity = np.zeros((self.N))
        fullnes = np.zeros((self.N))
        
        map_shape = np.array(self.obj_properties["map"].shape, dtype=float)
        map_shape = (map_shape/2.0).astype(int)
        obj_map = cv2.resize(self.obj_properties["map"], dsize = (map_shape[1], map_shape[0]))
        _, obj_map = cv2.threshold(obj_map, 1, 1, 0)

        proportion = np.zeros((self.N))

        for i in range(self.N): #np.where(self.particles[:, 2]*self.particles[:,3] >20)[0].tolist():
            
            # creating a map to every particle
            particle_map = observation_map[y[i]:y_end[i]+1, x[i]:x_end[i]+1]
            
            # if, from some reason, length or width is zero, the weight is zero and continue to next particle
            if l_x[i]==0 or l_y[i]==0:
                target_found[i] = 0
                rmse[i] = 1
                density[i] = 0
                proportion[i] = 0
                fullnes[i] = 0
                continue

            if target_box != []:
                # check the overlaping area between the target and the particle:
                xt, yt, l_xt, l_yt = target_box
                xt_end = xt + l_xt
                yt_end = yt +  l_yt
                left_upp = [max(xt, x[i]), max(yt, y[i])]
                rigt_dwn = [min(xt_end, x_end[i]), min(yt_end, y_end[i])]
                overlap_area = max(0, rigt_dwn[0]-left_upp[0]) * max(0, rigt_dwn[1]-left_upp[1])
                target_found[i] = min(1, overlap_area/float(l_xt*l_yt))
                # target_found[i] = 1 if bool_target(particle_map, th) else 0.2

            # fullnes is a measurement of how much of the bbox is full with the object. (avoid the "leakinng" aside of the object)
            [y_values, x_values] = np.where(particle_map > 0)
            if y_values.size == 0 or x_values.size == 0:
               fullnes[i] = 0
            else:
                f = float((np.max(x_values)-np.min(x_values)) * (np.max(y_values)-np.min(y_values))) / float(l_x[i] * l_y[i])
                fullnes[i] = f

            # rmse
            particle_map_resized = cv2.resize(particle_map, dsize=(map_shape[1], map_shape[0]))
            _, particle_map_resized = cv2.threshold(particle_map_resized, 1, 1, 0)
            rmse[i] = np.sum(np.power(obj_map-particle_map_resized, 2)) / 1

            # density
            d = float(np.sum(particle_map>0)/float(l_x[i] * l_y[i]))
            density[i] = d

            # relative size
            proportion[i] = float(l_x[i]) / float(l_y[i])
            #normalized_relative_size[i] = min(relative_size, relative_obj_size) / max(relative_size, relative_obj_size) if max(relative_size, relative_obj_size) else 0

        # ways to normalize the values:
        density_pdf = norm(self.obj_properties["density"], self.obj_properties["density"]/3)
        density_similarity = density_pdf.pdf(density) / density_pdf.pdf(self.obj_properties["density"])

        obj_proportion = float(self.obj_properties["width"]) / float(self.obj_properties["height"]) if self.obj_properties["height"] else 0
        proportion_pdf = norm(obj_proportion, obj_proportion/3)
        normalized_proportion = proportion_pdf.pdf(proportion) / proportion_pdf.pdf(obj_proportion)
        
        similarity = 1 - (rmse/float(map_shape[0]*map_shape[1]))
        #print similarity
        max_size = max(np.max(l_x*l_y), 1.0)
        normalized_size = l_x*l_y / max_size

        # self.weights = (0.2*normalized_size + 0.2*density_similarity + 0.6*similarity) * (density > 0.1) * (normalized_proportion > 0.5) * fullnes
        self.weights = similarity * (density>0.1) * (fullnes > 0.5) * (normalized_proportion > 0.5)
        if target_box != []:
            self.weights = self.weights * target_found


    def estimate(self):
        # the esimation step is estimating the location of the object in the observation. it takes the most probable particle.
        idx = np.argmax(self.weights) # estimation particle index
        if self.weights[idx] > 0.6:
            self.previous_state = self.state
            self.state = self.particles[idx, :]
            dstate = (self.previous_state - self.state)
            self.covs = [0.1*self.state[2], 0.1*self.state[3], min(0.1*self.state[2], 5), min(0.1*self.state[3], 5)]
            self.visibility = max(1, self.visibility+1)
            self.age += 1
            self.update_properties()
        else:
            # object not seen, because: 1. got out of the image, or 2. it was just noise.
            self.visibility = self.visibility-1 # min(-1, self.visibility-1)
            # self.covs = 2*self.covs
        if self.weights[idx] == 0:
    	    self.visibility = -1

        
