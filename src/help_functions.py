import numpy as np
import cv2
import os
import tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def window_smoothing(img, width):  # inputs: integral image, window's width
    I = cv2.integral(img, -1)
    I = I[1:img.shape[0]][1:img.shape[1]]
    rows, cols = I.shape
    up_left = I[0:rows - width, 0:cols - width]
    up_right = I[0:rows - width, width:cols]
    down_left = I[width:rows, 0:cols - width]
    down_right = I[width:rows, width:cols]

    window_map = down_right - down_left - up_right + up_left
    window_map = window_map / (width**2)

    #zero padding
    zp_r = 0.5 * (img.shape[0] - window_map.shape[0])
    zp_c = 0.5 * (img.shape[1] - window_map.shape[1])
    if zp_r != int(zp_r):
        zp_r_end = img.shape[0]-int(zp_r+1)
    else:
        zp_r_end = img.shape[0]-int(zp_r)
    zp_r = int(zp_r)

    if zp_c != int(zp_c):
        zp_c_end = img.shape[1]-int(zp_c +1)
    else:
        zp_c_end = img.shape[1]-int(zp_c)
    zp_c = int(zp_c)

    window_map_zp = np.zeros_like(img)
    
    r_end = np.min([window_map.shape[0], img.shape[0]])
    c_end = np.min([window_map.shape[1], img.shape[1]])

    window_map_zp[zp_r:zp_r_end, zp_c:zp_c_end] = window_map[0:r_end, 0:c_end]


    return window_map_zp.astype(np.uint8)


def find_center(img):
    # this method find the pcenter of the image: where the FLS is
    rows, cols = img.shape
    lastLine = np.array(img[rows - 1, :])
    l = np.where(lastLine > 0)
    l = l[0] # l is a tuple of list. extract the list.
    # if the last line is empty:
    i = 2
    while l.size<2: # while l deosn't have 2 points
        lastLine = np.array(img[rows - i, :])
        l = np.where(lastLine > 0)
        l = l[0]
        i = i+1

    sz = l.shape
    p1 = np.array([l[0], rows]).astype(np.float)
    p2 = np.array([l[sz[0]-1], rows]).astype(np.float)
    middle = int((p1[0] + p2[0]) / 2)
    col = np.where(img[:, middle] > 0)
    c = np.amax(col)
    p3 = np.array([middle, c]).astype(np.float)

    # method from: https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/

    x12 = p1[0] - p2[0]
    x13 = p1[0] - p3[0]
    y12 = p1[1] - p2[1]
    y13 = p1[1] - p3[1]
    y21 = p2[1] - p1[1]
    y31 = p3[1] - p1[1]
    x31 = p3[0] - p1[0]
    x21 = p2[0] - p1[0]

    sx13 = pow(p1[0], 2) - pow(p3[0], 2)
    sy13 = pow(p1[1], 2) - pow(p3[1], 2)
    sx21 = pow(p2[0], 2) - pow(p1[0], 2)
    sy21 = pow(p2[1], 2) - pow(p1[1], 2)

    f = ((sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) // (2 * (y31 * x12 - y21 * x13)))
    g = ((sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) // (2 * (x31 * y12 - x21 * y13)))
    c = (-pow(p1[0], 2) - pow(p1[1], 2) - 2 * g * p1[0] - 2 * f * p1[1])

    xc = int(-g)
    yc = int(-f)

    if np.isnan(xc) or np.isnan(yc):
         xc = p1[0]
         yc = p1[1]

    return [xc, yc]

def min_max_to_range(min, max, img):
    range_img = np.zeros_like(img)

    nrows, ncols = img.shape
    rows_mat = np.tile(range(ncols), (nrows, 1))
    cols_mat = np.tile(np.reshape(range(nrows), (nrows, 1)), ncols)
    pix_dist_mat = np.sqrt(np.power(cols_mat - nrows, 2) + np.power(rows_mat - ncols/2, 2))

    live_area = np.zeros_like(img)
    live_area[np.where(img>0.0)] = 1
    center_row = np.where(live_area[:, int(ncols/2)])
    range_min_idx, range_max_idx = np.max(center_row), np.min(center_row)
    pix_min, pix_max = pix_dist_mat[range_min_idx, int(ncols/2)], pix_dist_mat[range_max_idx, int(ncols/2)]

    a = (min - max) / (pix_min - pix_max)
    b = min - a*pix_min

    range_img = a*pix_dist_mat + b
    
    return range_img

def gen_range_img(start_range, stop_range, origin_col, origin_row, range_resolution , img):


    indices = np.where(img == [255])
    rows_mat = indices[0]
    cols_mat = indices[1]



    ranges = range_resolution * np.sqrt(np.power((cols_mat - origin_col), 2) +  np.power(rows_mat - origin_row, 2) )
    thetas = np.arctan2((-cols_mat + origin_col) , (-rows_mat + origin_row)) #- np.pi/2


    return ranges , thetas




def gen_range_img_oculus(start_range, stop_range , sonar_angle, img):


    sonar_rows,  sonar_cols  = img.shape
    range_resolution =  (stop_range - start_range) / float(sonar_rows)
    theta_resolution =   (sonar_angle /  float(sonar_cols)) 


    indices = np.where(img == [255])
    rows_mat = indices[0]
    cols_mat = indices[1]


    ranges =range_resolution * (float(sonar_rows) - rows_mat)
    thetas =theta_resolution * (cols_mat - float(sonar_cols) / 2)




    return ranges , thetas




def get_range(origin_col, origin_row, range_resolution , targets):


    if (targets): 
        rows_mat = np.array(zip(*targets)[1]) #targets[0][0]
        cols_mat = np.array(zip(*targets)[0]) #targets[0][1]
        #print(np.array(rows_mat))

        ranges = range_resolution * np.sqrt(np.power((cols_mat - origin_col), 2) +  np.power(rows_mat - origin_row, 2) )
        thetas = np.arctan2((-cols_mat + origin_col) , (-rows_mat + origin_row)) #- np.pi/2
  
        X = ranges *  np.cos(thetas) 
        Y = -ranges *  np.sin(thetas) 
        Z = np.zeros((len(X),), dtype=float) #- self.fls2ned_pos[2]
        
        points = np.column_stack([X,Y,Z])

        return points



def images2video(video_name, image_folder):

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        os.remove("output_images/" + image)

    cv2.destroyAllWindows()
    video.release()

def images2video2(video_name, images):
    
    height, width, layers = images[0].shape

    video = cv2.VideoWriter(video_name, 0, 5.0, (width,height))

    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()

# yevgeni added 

def transform_from_tf(xyz, rpy):
    """Convert a transform (xyz, rpy) of TransformHandler to numpy (xyz, rotation)."""
    v = np.array([[xyz[0], xyz[1], xyz[2]]]).T
    r = tf.transformations.euler_matrix(rpy[0], rpy[1], rpy[2])[:3, :3]
    return v, r
    #inverse transform
    #return -r.T.dot(v), r.T
    #return -v, r.T

 
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)) 


