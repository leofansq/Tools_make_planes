# Make Planes #

AVOD needs the planes file to provide ground plane information, but the official planes generation tool has not yet been provided, which brings great difficulty to the test work. This project is used to generate planes files especially for AVOD testing. 

## Functions ##
#### Generate the planes files for the KITTI testing data or your own data ####

* lidar4to3(input_file_path, output_file_path): Convert the KITTI lidar points from Nx4 shape to Nx3 shape.

* cal_planes(input_file_path="./points/", output_file_path="./planes/"): Using Ransac in PyntCloud to find the groud plane.

* lidar_point_to_img_calib(point, Tr, R0, P2): Convert lidar points to the camera.

* calib_at(file_path, file_index): Return the calib sequence.

* load_kitti_calib(file_path, file_index): Load projection matrix.


## Input ##

* calib
* velodyne(Nx4)

## Output ##

* points(Nx3)
* planes

## Getting Started ##

* The  project is based on PyntCloud. You need to install it first.

    ```git clone https://github.com/daavoo/pyntcloud.git```

	```pip install -e pyntcloud```
    
    > It seems like the PyntCLoud is updated. This code may not work for the new version. **So I Upload The Old Version In This Repo.**

* Set the path of your dataset in [main.py](main.py). 

    You can also set the specified output path for the generated Points(Nx3) files and the Planes files. **Note that the input path of *cal_planes()* needs to be consistent with the output path of *lidar4to3()*.** The default output path for the Points file is *'./points/'*, and the default output path for the Planes file is *'./planes/'*.

* Run the *main.py* to start generating.

    ```python main.py```

* For the KITTI dataset, the error data self-checking function is included in the *cal_planes()* function. Plane data with a height less than 1.3 meters or greater than 2.0 meters is determined as erroneous data and replaced with default data (0, -1, 0, 1.65). **The decision criteria and default values can be modified in the *cal_planes()* function**. For erroneous data, its name and origin plane data will be stored in the ***error.log*** file.

    > Due to the principle of the generation of planes, there may be subtle differences in the generated planes data each time.

## Supplementary Information ##

* More information about [AVOD](https://github.com/kujason/avod)

* More information about [PyntCloud](https://pyntcloud.readthedocs.io/en/latest/index.html)
