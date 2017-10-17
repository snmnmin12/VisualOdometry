### Requirement
* OpenCV 3.0
* Kitti data or you own data
* Tested with Cmake on my mac

### Introduction

This implementation is for 3D-2D visual odometry, you need to have stereo image pairs to generate 3D points for feature points, and then track the feature points to the new coming frame, and then recover the pose of the newly joined frame as the current pose for the new frame.

The feature points becomes less as tracking processing going on forward. Here, my strategy is quite simple. Drop all the feature points from the old frame and extract new frames for the new image pair and 3D points. Since the transformation matrix R is available and now you can make simple transformation to transform the new 3D points back to the coordinate frame based on the initial frame 0. In this way, all the 3D point coordinates are based on the frame 0 and all poses are according to frame 0.

This source code is easy to understand and follows the steps:

Initialize feature points, 3D points --> Track feature points and recover pose --> create new feature points, 3D points --> go back to track again!.

If you wan to know how to add bundle adjustment for optimization of poses, please refer to the version 2 on how to add bundle adjustment.


[This tutorial](https://github.com/gaoxiang12/rgbd-slam-tutorial-gx) and 
[This blog](http://www.cnblogs.com/gaoxiang12/tag/%E8%A7%86%E8%A7%89%E9%87%8C%E7%A8%8B%E8%AE%A1/) for explanation.