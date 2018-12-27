## Make Planes ##

AVOD needs the planes file to provide ground plane information, but the official planes generation tool has not yet been provided, which brings great difficulty to the test work. This project is used to generate planes files especially for AVOD testing. 

### Functions ###
 
* Generate a planes file of the KITTI testing part.
* generate planes for specific files. (Still have some problems)

### Input ###

* calib
* velodyne

### Output ###

* planes

### Getting Started ###

* The  project is based on PyntCloud. You need to install it first.

    ```git clone https://github.com/daavoo/pyntcloud.git```

	```pip install -e pyntcloud```

* Adjust the running mode in [main.py](main.py). You can choose to generate planes for specific files, or you can generate KITTI's planes in batches.

* Set parameters in [make_planes.py](make_planes.py). You need to set the input and output file path.

* Run the main.py to start generating.

### Supplementary Information ###

* More information about [AVOD](https://github.com/kujason/avod)

* More information about [PyntCloud](https://pyntcloud.readthedocs.io/en/latest/index.html)

### TO DO ###

- [] Fix the problem for generate planes for specific files.
