# This project is used to generate planes files
# By leofansq
#################################################################

import make_planes as mp

def main():

    print("start")

    # Specific project processing
    #mp.lidar4to3()
    #mp.cau_planes()

    # KITTI Dataset processing
    mp.lidar4to3_kitti()
    mp.cau_planes_kitti()

    pass

if __name__ == '__main__':
    main()