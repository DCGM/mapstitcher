mapstitcher
===========

Allows to stitch arbitrary large images.

Installation
------------

In command line run:

$ git clone https://github.com/mimunzar/mapstitcher mapstitcher
$ cd mapstitcher
$ python3 -m venv ${PATH_TO_VENV}
$ source ${PATH_TO_VENV}/bin/activate
$ pip install --upgrade pip && pip install -r requirements.txt
$ sudo apt update
$ sudo apt install libopenjp2-tools

Running as CLI
--------------

To run the image stitching process, use the following command:

```bash
python image_stitch_batch.py --list list.txt

list.txt contains the configuration for images, and stitching rows. Ensure it follows the required format detailed below.

Parameters:
--list 			optional, File and processing list
--path 			optional, Path to folder with name-coded images
--optimization-model 	optional, default='affine', Optimization model homography or affine
--matching-algorithm optional, default='loftr', Matching algorithm loftr or sift
--loftr-model optional, default='outdoor', Model for LOFTR - outdoor or indoor
--flow-alg 		optional, default='raft', Optical Flow algorithm cv (not working at the moment) or raft
--subsample-flow 	optional, default=2.0, Subsample flow
--vram-size		optional, default=8.0, GPU VRAM size in GB, more can speed-up raft optflow computation 
--max-matches		optional, default=800, Maximum number of matches per image pair (for optimization)
--output		optional, default='result.jp2', Output file, if .jp2 extension is used, openjp2-tools will be used to convert the file (no compression) 
--silent		optional, default=False, Kill any console output

Input As Name-Coded Files
-------------------------
The input directory can contain any number of name-position coded images.
The program will create neighbor pairs from the file names. For example directory can contain files:
image0_0_0.png
image1_0_1.png
image2_0_2.png
image3_1_0.png
image4_1_1.png
image5_1_2.png

where the first index represents row and the second index represents column.
So the intended image configuration is as follows:
image0 image1 image2
image3 image4 image5

Input List Format
-----------------
The input list file (list.txt) should be structured as follows:

- images
0 path_to_image_0
1 path_to_image_1
2 path_to_image_2
...
N path_to_image_N
- rows
0 1
2 N

where 
'images' section contains an enumerated list of images to be stitched, starting from 0.

'rows' section defines the structure of the image stitching. Each line represents a row of images.

Example Input List Configurations
---------------------------------
For stitching four images left-to-right in a single row:
- images
0 path/image0.png
1 path/image1.png
2 path/image2.png
3 path/image3.png
- rows
0 1 2 3

For stitching six images with three on the top row and three on the bottom row (assuming overlap right and bottom):
- images
0 path/image0.png
1 path/image1.png
2 path/image2.png
3 path/image3.png
4 path/image4.png
5 path/image5.png
- rows
0 1 2
3 4 5

Notes
-----
Ensure all paths in the images section are correct and point to the respective images.
Each section (images, rows) must start with a - followed by the section name.
The list is case-sensitive, and section names should be written exactly as shown (images, rows).
You can find simple example in test_data directory

Running as Docker service
-------------------------
All files required for building and using docker service are located in 'docker' folder.
In the file docker-compose.yaml it is necessary to set up paths to inputs and outputs.

Build docker:
$ docker compose -f docker-compose.yaml build

To use docker service one of the following commands can be used
$ docker compose up mapstitcher
$ docker compose up mapstitcher_service

'mapstitcher' processes input directory and exits. 
'mapstitcher_service' monitors input directory and processes any new folder it finds. 
To avoid processing folder while data is being transfered to it, use suffix "_incomplete"
in the folder name. Once the data is ready, rename the folder.
Folders with suffices "_incomplete" or "_done" are ignored.
