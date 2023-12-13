import numpy as np
from help_functions import find_center


def create_euler_matrices(rpy_var):
    [r, p, y] = np.deg2rad(rpy_var)

    rx = [[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]]
    ry = [[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]]
    rz = [[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]]

    R = np.matmul(rx, np.matmul(ry, rz))  # r = rx x ry x rz

    return R


def motion_model(state, center, range_img, position_var, orientation_var):
    # INPUT: target's distance from the sonar and change in the vehicle's center position
    # OUTPUT: the expected position in the image

    # NOTE: in all angle computations, theta is the angle between the y-axis to the line between fls-assumed-position and the target

    # distance between sonar and vehicle's center
    d_xyz = np.array([0.5275, 0, 0.109]) # [m]

    r1_sonar = range_img[state[1], state[0]] # [m]
    theta1_sonar = np.arctan(float(state[0]-center[0])/float(state[1]-center[1])) # [rad]

    # [r, theta] to [x, y]
    x1_sonar = r1_sonar * np.sin(theta1_sonar) # [m]
    y1_sonar = r1_sonar * np.cos(theta1_sonar) # [m]
    # "old" position from sonar to "old" position from vehicle
    xyz1_vehicle = np.array([x1_sonar, y1_sonar, 0]) + d_xyz # [m]

    # change in the vehicle center position
    r = create_euler_matrices(orientation_var)
    xyz2_vehicle = position_var + r.dot(xyz1_vehicle.T) # [m]

    # position from vehicle to position from sonar
    xyz2_sonar = xyz2_vehicle - d_xyz # [m]

    # transform to angle and radius
    r2_sonar = np.sqrt(np.power(xyz2_sonar[0], 2) + np.power(xyz2_sonar[1], 2)) # [m]
    theta2_sonar = np.arctan((xyz2_sonar[0])/(xyz2_sonar[1])) # [rad]
    # using range_img transform radius to radius in image
    pixels_list = np.where(np.round(range_img, 1)==np.round(r2_sonar, 1)) # [pix] [y, x]
    x_list = pixels_list[1]
    y_list = pixels_list[0]
    if pixels_list[0].size == 0:
        return state[0], state[1], xyz2_sonar[2]
    pixels_list_thetas = np.zeros_like(x_list, dtype=float)
    yb = np.where(y_list != center[1])
    pixels_list_thetas[yb] = np.arctan((x_list[yb].astype(float)-float(center[0])) / (y_list[yb].astype(float)-float(center[1]))) # [rad], pix
    pixels_list_thetas[np.where((y_list == center[1]) & (x_list > center[0]))]  = np.pi
    dtheta = abs(pixels_list_thetas - theta2_sonar)
    [x, y] = x_list[np.argmin(dtheta)], y_list[np.argmin(dtheta)]
    #if np.linalg.norm(np.array([x,y])-state[:2]) > 5:
    #    return x, y , 0

    # returning  in pixels and z in meters
    return x, y, xyz2_sonar[2] 
    
