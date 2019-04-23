import make_planes as mp

# Path of the dataset, for KITTI is "/home/xxxxx/KITTI/training/"
DATA_FILE_PATH = "./training/"

print ("Start", DATA_FILE_PATH)

# lidar4to3(input_file_path, output_file_path="./points/")
mp.lidar4to3(DATA_FILE_PATH)

# cal_planes(input_file_path="./points/", output_file_path="./planes/")
mp.cal_planes()