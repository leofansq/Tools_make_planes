"""
@author:Leofansq

MAIN FUNCTION:

* lidar4to3(input_file_path, output_file_path): Convert the KITTI lidar points from Nx4 shape to Nx3 shape.

* cal_planes(input_file_path="./points/", output_file_path="./planes/"): Using Ransac in PyntCloud to find the groud plane.

* lidar_point_to_img_calib(point, Tr, R0, P2): Convert lidar points to the camera.

* calib_at(file_path, file_index): Return the calib sequence.

* load_kitti_calib(file_path, file_index): Load projection matrix.
"""
from pyntcloud import PyntCloud
import numpy as np

import os
import time

import scipy.linalg
from tqdm import tqdm

def lidar4to3(input_file_path, output_file_path="./points/"):
    """
    Convert the KITTI lidar points from Nx4 shape to Nx3 shape
    i.e. remove the reflectivity
    
    Parameters:
    input_file_path: the path of the dataset, for KITTI is "/home/xxxxx/KITTI/training/"
    output_file_path: the dir_path to save the rebuild points cloud files
    """
    print ("-----   Convert the LiDAR points from Nx4 shape to Nx3 shape   -----")
    file_path = input_file_path + "velodyne/"
    for file_name in tqdm(os.listdir(file_path)):
        #print("Processing: ", file_name)
        points = np.fromfile(file_path + file_name, dtype=np.float32)
        #print(np.shape(points))
        points = points.reshape((-1, 4))
        points = points[:, :3]

        calib = calib_at(input_file_path, file_name[:-4])

        # points input: nx3; points output: nx3;
        points = lidar_point_to_img_calib(points, calib[3], calib[2], calib[0])
        points = points.astype(np.float32)

        file_output_name = output_file_path + file_name

        points.tofile(file_output_name)
    print()

def cal_planes(input_file_path="./points/", output_file_path="./planes/"):
    """
    Using Ransac in PyntCloud to find the groud plane.
    Groud plane parameters (A, B, C, D) for Ax+By+Cz+D=0.
    Note the lidar points have transformed to the camera coordinate.

    Parameters:
    input_file_path: the path of the points files, the shape of the points should be Nx3
    output_file_path: the path to save the plane files
    """
    print ("----------   Calculating the planes   ----------")
    f_error = open("error.log", "w")
    error_cnt = 0
    error_flag = False
    for file_name in tqdm(os.listdir(input_file_path)):
        #print ("Processing: ", file_name)
        cloud = PyntCloud.from_file(input_file_path + file_name)
        cloud.points = cloud.points[cloud.points["y"] > 1]

        is_floor = cloud.add_scalar_field("plane_fit", n_inliers_to_stop=len(cloud.points) / 20, max_dist=0.001, max_iterations=500)

        cloud.points = cloud.points[cloud.points[is_floor] > 0]
        data = np.array(cloud.points)

        # best-fit linear plane : Z = C[0] * X + C[1] * Y + C[2]
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

        normal = np.array([C[0], C[1], 1, C[2]])
        normal = - normal / normal[1]
        #print(normal)

        # Check if the result is almost the groud plane.
        # if the result is right, parameter B should be nearly 1 when the D is the height of the camera.
        # if the result is not right, wirte the default value for KITTI
        if (normal[3] > 2.0 or normal[3] < 1.3) :
            #print("error_result")
            error_flag = True
            error_cnt += 1
            f_error.write(file_name[:-4] + ".txt    " + str(normal[0]) + " " + str(normal[1]) + " " + str(normal[2]) + " " + str(normal[3]) + "\n")
            
            str_normal = "0.0" + " " + "-1.0" + " " + "0.0" + " " + "1.65"
        else:
            str_normal = str(normal[0]) + " " + str(normal[1]) + " " + str(normal[2]) + " " + str(normal[3])

        plane_file_name = output_file_path + file_name[:-4] + ".txt"
        f = open(plane_file_name, "w")

        f.write("# Plane\n")
        f.write("Width 4\n")
        f.write("Height 1\n")
        f.write(str_normal)

        f.close()
    f_error.close()
    if error_flag:
        print ("\n There are ", error_cnt, " planes results may not be right! \n The files' name is saved in error.log")

def lidar_point_to_img_calib(point, Tr, R0, P2):
    """
    Convert lidar points to the camera

    Parameters:
    point: points with shape Nx3;

    Return:
    point with shape NX3 (N is the number of the points)
    output = R0*Tr*point
    """
    P2 = P2.reshape((3, 4))
    R0 = R0.reshape((4, 3))
    Tr = Tr.reshape((3, 4))

    T = np.zeros((1,4))
    T[0,3] = 1

    P2 = np.vstack((P2, T))
    Tr = np.vstack((Tr, T))

    T2 = np.zeros((4,1))
    T2[3,0] = 1
    R0 = np.hstack((R0, T2))

    assert Tr.shape == (4, 4)
    assert R0.shape == (4, 4)
    assert P2.shape == (4, 4)

    point = point.transpose((1, 0))

    point = np.vstack((point, np.ones(point.shape[1])))

    mat = np.dot(R0, Tr)
    img_cor = np.dot(mat, point)

    # img_cor = np.dot(Tr, point)
    
    img_cor = img_cor.transpose((1, 0))

    img_cor = img_cor[:, :3]

    return img_cor

def calib_at(file_path, file_index):
    """
    Return the calib sequence.

    Parameters:
    file_path: the path of the dataset, for KITTI is "/home/xxxxx/KITTI/training/"
    file_index: the index of the calib file

    Return:
    the calib sequence
    """
    calib_ori = load_kitti_calib(file_path, file_index)
    calib = np.zeros((4, 12))
    calib[0,:] = calib_ori['P2'].reshape(12)
    calib[1,:] = calib_ori['P3'].reshape(12)
    calib[2,:9] = calib_ori['R0'].reshape(9)
    calib[3,:] = calib_ori['Tr_velo2cam'].reshape(12)

    return calib

def load_kitti_calib(file_path, file_index):
    """
    Load projection matrix

    Parameters:
    file_path: the path of the dataset, for KITTI is "/home/xxxxx/KITTI/training/"
    file_index: the index of the calib file

    Return:
    the projection matrix
    """
    calib_file_name = os.path.join(file_path + "calib/" + file_index + ".txt")

    with open(calib_file_name) as calib_file:
        lines = calib_file.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}
