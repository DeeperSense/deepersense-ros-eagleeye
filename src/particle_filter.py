
import numpy as np
from numpy.random import uniform
import cv2
import matplotlib.pyplot as plt
from help_functions import find_center
import munkres
from scipy.ndimage import binary_fill_holes
from get_objects import get_objects

# this class is implementing the partcile filter tracking algorithm
# more reading about how particles filter algorithm is working:
# Wikipedia. and more good explanation about one-dimension-point-object tracking in this video: https://www.youtube.com/watch?v=aUkBa1zMKv4

class ParticleFilter:
	# init the object. This method initiating the variables I'll need in the algorithm
    def __init__(self):

        self.obj_size = 30
	    # objects list (estimated on the estimate method) is described by bounding rectangle, this described by x, y of the left-up point, width and height:
        self.objects_list = np.empty((0, 6))  # x, y, w, h, idx, detections_num
	    # the objects tracked is indexed to track what object I'm tracking. start from index=1
        self.object_index = 1
	    # color list specify a color for every track. Organized by the indexes of the tracks
        self.color_list = np.zeros((1, 3))

    def get_img(self, img, index):
        self.img = img

        if index==1:
            self.width, self.height = self.img.shape
            # N is the number of particles through all the code
            self.N = int(self.width * self.height / self.obj_size)
            # the sonar omage has a lot of "dead area" where there is nothing. alive area is black&white image: black-dead area, white-alive area
            self.alive_area = np.zeros_like(self.img, dtype=float)
            self.alive_area[np.where(self.img > 0.0)] = 1.0
            # the particles list initiated with uniform distribution on the live area.
            self.particles = self.create_uniform_particles(img)
	        # weight for every particle initiated with 1.
            self.weights = np.ones((1, self.N))
            # birth_model is a model predicting where new objects will appear. noise model is just noise model.Both described by map in the same size of the original image.
            self.birth_model = self.create_birth_model()
            self.noise_model = self.create_noise_model()

 	        # obj_img is image of objects found in estimate
            self.obj_img = np.zeros_like(self.img)
 	        # posterior is the predicted map of objects for the next image, create in predict method
            self.posterior = np.zeros_like(self.img)


    # for initiating the partciles position uniformly:
    def create_uniform_particles(self, img):
        pixel_list = np.where(img>0.0)
        pixel_list = np.array(pixel_list)
        uniform_array = uniform(0, pixel_list[0].size, size=self.N)
        uniform_array = uniform_array.astype(np.int)
        particles = pixel_list[:, uniform_array]
        return particles.T

    # birth model is assunmes that new objects will apear only on the boundaries of the image
    # OUTPUT: map in the same shape of the original image, valued with the probability to find there object in the next image, by the birth model.
    def create_birth_model(self):
        birth_model = cv2.GaussianBlur(self.alive_area, (29, 29), 20, 20)  # TODO: change the numbers
        birth_model = np.ones_like(birth_model, dtype=float) - birth_model
        birth_model[np.where(birth_model == 1)] = 0
        birth_model[np.where((birth_model < 0.05) & (birth_model > 0.0))] = 0.05

        return birth_model

    # noise model assumes noise close to the FLS (rely on the images I saw)
    # OUTPUT: map in the same shape of the original image, valued with the probability to find there noise in the next image, by the noise model.
    def create_noise_model(self):
        noise_model = np.zeros_like(self.img, dtype=float)
        [xc, yc] = find_center(self.img)

        radius = 100
        rows, cols = self.img.shape
        for y in range(rows - radius, rows):
            for x in range(xc - radius, xc + radius):
                r = np.sqrt(((xc - x) ** 2) + ((yc - y) ** 2))
                if r <= radius and self.alive_area[y, x] == 1:
                    noise_model[y, x] = 1

        return noise_model

    # help function: creates normal distribution around the objects found. used in motion_model
    def norm_distribute(self, loc=[0, 0, 1, 1], scale=1):
        loc = [float(i) for i in loc]
        size_x = loc[2]
        size_y = loc[3]
        sigma_x = (loc[2] / max(loc[2], loc[3])) * 10
        sigma_y = (loc[3] / max(loc[2], loc[3])) * 10
        x = np.linspace(-10, 10, size_x)
        y = np.linspace(-10, 10, size_y)
        x, y = np.meshgrid(x, y)
        if sigma_x==0 or sigma_y==0:
            z = 0
        else:
            z = 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-(x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)))
            z = z / np.max(z)
        
        return z

    # motion_model get the objects estimated, and assumes that in the next image, we will see the same objects close to where we saw them in this image.
    # OUTPUT: map in the same shape of the original image, valued with the probability to find there object in the next image, by the motion model.
    # this function can be improved if it gets the notion of the SPARUS. Now it's just using normal distribution around the objects' positions.
    def apply_motion_model(self):
        moved_image = np.zeros_like(self.img)
        for obj in self.objects_list:
            x = max(0, obj[0] - 5)
            y = max(0, obj[1] - 5)
            w = obj[2] + 10
            if x+w >= self.img.shape[1]:
                w = self.img.shape[1] - x - 1
            h = obj[3] + 10
            if y+h >= self.img.shape[0]:
                h = self.img.shape[0] - y - 1
            moved_image[y:y+h, x:x+w] = self.norm_distribute([x, y, w, h])
        moved_image[np.where(moved_image > 1)] = 1
        return moved_image

    # update takes the image and updating the weight of every pixel.
    # the new weight is the value of the ROIs image (in the ROIs image (called here mask) every pixel is valued by the probability to find there object)
    def update(self, mask):
        self.img = mask
        self.weights = mask[self.particles[:, 0], self.particles[:, 1]]

    # estimate method estimate, by the particles and their weights, where there are objects in this image.
    # TODO: convert objects to NED coordinates.
    def estimate(self):
        objects_list = np.empty((0, 4))
        weights_th = 120
        dilate_kernel_size = 15

	    # create image of zeros and apply to the particles' positions the weights of these particles
        weights_img = np.zeros_like(self.img)
        weights = (self.weights/self.weights.max())*255
        particles = self.particles[np.where(weights > weights_th)]
        weights = weights[np.where(weights > weights_th)]
        weights_img[particles[:, 0], particles[:, 1]] = weights

	    # because the objects are not point-objects, can be that in one object will be some particles. so the particles image here is converted to map of objects:
	    # by dilating every pixel, the close particles with high weights will connect to one object
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
        weights_img = cv2.dilate(weights_img, kernel)
        weights_img = weights_img.astype(np.uint8)
        weights_img = binary_fill_holes(weights_img)
        weights_img = weights_img.astype(np.uint8)

	    # extracting from the map created the list of objects:
        contours, hierarchy = cv2.findContours(weights_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[cnt])
            objects_list = np.vstack((objects_list, [x, y, w, h]))
        objects_list = objects_list.astype(np.int)
        self.obj_img = weights_img

        return objects_list

    # predict is creating a map (same shape as original image) of predicted assumed positions of objects in the next image.
    # every pixel valued with the probability to find there object in the next image, by the motion model.
    def predict(self):
        motion_model = self.apply_motion_model()
        self.posterior = self.posterior + self.birth_model + motion_model - self.noise_model
        self.posterior[np.where(self.posterior < 0)] = 0
        self.posterior /= self.posterior.sum()
        # posterior[np.where(posterior > 0)] = 1
        pairs = np.indices(dimensions=self.img.shape).T
        inds = np.random.choice(np.arange(self.img.size), p=self.posterior.T.reshape(-1), size=self.N, replace=True)
        # self.particles = self.create_uniform_particles(posterior)
        self.particles = pairs.reshape(-1, 2)[inds]
        self.weights = np.ones((1, self.N))

    # after estimating the new objects in this image, we want to associate every object with a track.
    # the method using munkres (hungarian) algo to associate (you can read about munkres here: http://software.clapper.org/munkres/
    def objects_association(self, objects_list):
        n_tracks = self.objects_list.shape[0]
        n_detections = objects_list.shape[0]
        n = max(n_tracks, n_detections)
	    # unfortunately, munkres get only squared cost matrices, so the cost matrix is padded with high value.
        cost = np.ones((n, n)) * 200
        size_dif_matrix = cost.copy()
        distance_m = cost.copy()
	    # assigning to every not-padding position in the cost matrice a cost for associate the i-th track with the j-th object
        for i in range(n_tracks):
            for j in range(n_detections):
		    # the cost here is valued by the difference between the size and the position of the object and the track.
		    # can be optimized by other parameters (tried: moments, hu_moments and more). maybe need to be machine-learned...
                prev_bbox = self.objects_list[i] # the track
                bbox = objects_list[j] # the object found in this image
                distance = np.sqrt(((prev_bbox[0] - bbox[0]) ** 2) + ((prev_bbox[1] - bbox[1]) ** 2))
                distance_factor = 2 * distance / (prev_bbox[2] * prev_bbox[3])
                size_dif = np.sqrt(((bbox[2] - prev_bbox[2]) ** 2) + ((bbox[3] - prev_bbox[3]) ** 2))
                size_dif_factor = 30 * size_dif / (prev_bbox[2] * prev_bbox[3])
                cost[i][j] = (size_dif_factor + distance_factor) / 2
                size_dif_matrix[i][j] = size_dif_factor
                distance_m[i][j] = distance_factor

	    # get assignments from munkres. assignments is an array of pairs of track number and object number, that associating them is minimizing the cost:
        cost1 = np.copy(cost)
        m = munkres.Munkres()
        assignments = munkres.Munkres.compute(m, cost_matrix=cost1)

	    # for every assignment, should decide if to associate track with object assignmet to it, or not, by the cost.
        max_cost = 1.5  # max cost to implement the assignment
        unassigned_idx = np.empty((0, 1), dtype=int)
        
        for i in range(len(assignments)):
            prev_obj_idx = assignments[i][0] # index of track
            obj_idx = assignments[i][1] # index of object found in this image

            # object index is out of bound of object_list (can happen because of padding)
            if obj_idx >= objects_list.shape[0]:
                if prev_obj_idx < self.objects_list.shape[0]:  # if the track is not out of bound of tracks list
                    if self.objects_list[prev_obj_idx][5] < 0:  # the 6-th element is the number of detections. if it didn't detect on the previous frame:
                        self.objects_list[prev_obj_idx][5] -= 1
                    else:                                       # if it's first time the track not detected:
                        self.objects_list[prev_obj_idx][5] = -1
                unassigned_idx = np.vstack((unassigned_idx, i)) # add to unassigned_idx the number of the assignment that didn't implement
                continue
            
	        # track index is out of bound of track_list (can happen because of padding)
            if prev_obj_idx >= self.objects_list.shape[0]:
		        # if the detection is in the objects_list, start for it new track:
                if obj_idx < objects_list.shape[0]:
                    new_obj = np.concatenate((objects_list[obj_idx], [self.object_index, 0]))
                    self.objects_list = np.vstack((self.objects_list, new_obj.tolist()))
                    self.object_index += 1
                    new_color = [np.random.rand(), np.random.rand(), np.random.rand()]
                    self.color_list = np.vstack((self.color_list, new_color))
                    continue
        
	    # if the cost is above the max_cost, the assignment is not implemented:
	    # the track's detections number is -1 (as it didn't detect on this frame), and the object getting a new track
            if cost[prev_obj_idx][obj_idx] >= max_cost:
                new_obj = np.concatenate((objects_list[obj_idx], [self.object_index, 0]))
                self.objects_list = np.vstack((self.objects_list, new_obj.tolist()))
                self.object_index += 1
                new_color = [np.random.rand(), np.random.rand(), np.random.rand()]
                self.color_list = np.vstack((self.color_list, new_color))
                unassigned_idx = np.vstack((unassigned_idx, i))  # index from assignments
                if self.objects_list[prev_obj_idx][5] < 0:
                    self.objects_list[prev_obj_idx][5] -= 1
                elif prev_obj_idx != self.objects_list.shape[0]-1:
                    if self.objects_list[prev_obj_idx][5] < 0:
                        self.objects_list[prev_obj_idx][5] -= 1
                    else:
                        self.objects_list[prev_obj_idx][5] = -1
	        # if the cost is under the max_cost, associate the object with the track, so update the track to the detections parameter, and increase the detections number:
            else:
                idx = self.objects_list[prev_obj_idx][4]
                detections_num = self.objects_list[prev_obj_idx][5]
                if detections_num < 0:
                    detections_num = 0
                else:
                    detections_num += 1
                self.objects_list[prev_obj_idx] = np.concatenate((objects_list[obj_idx], [idx, detections_num]))

	    # for every assignement not implemented, check if the track didn't detect for delete_number, and add to objects_to_delete:
        delete_number = -10
        num_unassigned, _ = unassigned_idx.shape
        objects_to_delete = np.empty((0, 1))
        for i in range(num_unassigned):
            j = unassigned_idx[num_unassigned - i - 1]
            prev = assignments[j[0]][0]
            if self.objects_list[prev][5] < delete_number:
                objects_to_delete = np.vstack((objects_to_delete, prev))
            del assignments[j[0]]

	    # delete the objects from the list:
        self.objects_list = np.delete(self.objects_list, objects_to_delete, 0)
        self.objects_list = self.objects_list.astype(int)

        # delete objects contained in other objects:
        contained = []
        blur_size = np.array([5, 5])
        for ind1 in range(self.objects_list.shape[0]):
            o1 = self.objects_list[ind1, :]
            for ind2 in range(self.objects_list.shape[0]):
                o2 = self.objects_list[ind2, :]
                if (o1 == o2).all():
                    continue
                if (o2[0:2] > o1[0:2] - blur_size).all():
                    if ((o2[0:2]+o2[2:4]) - blur_size < (o1[0:2]+o1[2:4])).all():
                        contained.append(ind2)
                        if o2[5] > o1[5]:
                            self.objects_list[ind1, 4:] = o2[4:]
        contained = np.array(contained)
        self.objects_list = np.delete(self.objects_list, contained, 0)

    def track_it(self, img, mask, range_img, index, snr=0.7):
        self.get_img(img, index)
        
        # updating the tracks with the info from the new image
        self.update(mask)

		# estimating the objects in this image
		# TODO: divide objects_list to obstacles and station
        object_list = self.estimate()

		# associate the objects estimated to the tracks from the previous images
        self.objects_association(object_list)

		# predict the positions of objects in the next image, using motion model and birth (of new objects) model
        self.predict()

        return self.objects_list, self.color_list
