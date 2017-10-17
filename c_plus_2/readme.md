### Requirement
* OpenCV 3.0
* Kitti data or you own data
* Tested with Cmake on my mac
* **g2o** for optimization

### Introduction

In the **C++** version 2, I added bundle adjustment to perform optimization in the backend and I choose to use the graph optimization package **g2o** for bundle adjustment. In order to install  **g2o** in you machine, you have to refer [this](https://github.com/RainerKuemmerle/g2o) for additional information. 

The basic idea is to optimize both the pose and 3D landmark and minimize the reprojection error between these 3D points, 2D feature points with recovered poses. In my implementation, the world 3D points-landmarks are stored and each frame only observe a subset of 3D points from these 3D points, and 3D landmark points keeps adding when new frame comes. 

Compare with version 1, I have added two files for bundle adjustment, you need to refer to g2o examples from internet to understand on how to use it. Besides, I have added the data structure to hold all landmarks obtained in each frame and remember frame key information with a struct **FrameInfo**, basically speaking, it hold the feature points information for each frame, camera pose information rotation vector, translation vector, and 3D object coordinate. Since I already have all the world points information, I only need to store the index of the world points corresponding to the current feature points.

```
struct FrameInfo{
	std::vector<cv::Point2f> features2d;
	std::vector<int> obj_indexes;
	cv::Mat rvec;
	cv::Mat tvec;
};
```
In addition to that, I have to create new feature points and new 3D landmarks for new frames, but I want to make sure new feature points not close to the old feature points from existing frame. Remember that, I have tracked feature points from frame to frame, so current frame has some feature points, and newly created feature points should not be the same with old feature points, then I added the remove duplicate function to remove duplicate feature points.

```
vector<int> VO::removeDuplicate(const vector<Point2f>& baseFeaturePoints, const vector<Point2f>& newfeaturePoints, 
	const vector<int>& mask, int radius=10)
```

If you want to know the details, you can proceed to the source files for information.

The bundle adjustment files are the basic configuration for graph optimization with **g2o** and some matrix transformation function and it is easy to understand.

Or you can refer to this github tutorial for further usage of g2o on bundle adjustment
[This tutorial](https://github.com/gaoxiang12/rgbd-slam-tutorial-gx) and 
[This blog](http://www.cnblogs.com/gaoxiang12/tag/%E8%A7%86%E8%A7%89%E9%87%8C%E7%A8%8B%E8%AE%A1/) for explanation.