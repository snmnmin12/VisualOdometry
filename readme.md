### Requirement
* Python 3.5
* Numpy
* OpenCv

### Introduction

There are currently two files in the src folder,  one is for the 2D-2D mono Visual Odometry and the other one is for 3D reconstruction, the third files is for 3D-2D visual odometry, so you have to initialize the 3D points first as landmarks, and then use landmarks and p3p to find the pose of camera and keep adding new landmarks as time goes.


### How to run
In the src files, you can simply run, here python is python3

```
python filename
```
But you need to set your data path to the correct data path now.

For the c++ files, you have to run below to get it running. Again, you have to modify the data path in the main file.

```
mkdir build
cd build
cmake ..
make
```
The dataset tested is from part of Kitty odometry data set.

![](img/1.png)
![](img/2.png)
![](img/4.png)
![](img/disparity.jpeg)
![](img/map.png)

### Reference
1. [Fall 2017 - Vision Algorithms for Mobile Robotics](http://rpg.ifi.uzh.ch/teaching.html)
2. [Monocular Visual Odometry using OpenCV](http://avisingh599.github.io/vision/monocular-vo/)| Avi Singh
