
#ifndef VO_H
#define VO_H


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <assert.h> 
// #include "opencv2/core/eigen.hpp"

// struct FrameInfo{

// 	std::vector<cv::Point2f> features2d;
// 	std::vector<cv::Point3f> features3d;
// 	cv::Mat rvec;
// 	cv::Mat tvec;
// };

struct FrameInfo{

	std::vector<cv::Point2f> features2d;
	std::vector<int> obj_indexes;
	// std::vector<cv::Point3f> features3d;
	cv::Mat rvec;
	cv::Mat tvec;
};

class VO {

public:

	//constructor
	VO(){}
	//defautl destructor
	virtual ~VO(){}
	VO(const cv::Mat& _k, float _baseline, const std::string&, const std::string&);

	//input is the feature points 1 and feature points 2 in each image, output is the 3D points, stereo images
	std::vector<cv::Point3f> get3D_Points(const std::vector<cv::Point2f>& feature_p1, const std::vector<cv::Point2f>& feature_p2) const ;

	//input  is two stere image pair, output is the feature points of image1 and generated 3D points of landmarks
	void extract_keypoints_surf(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::Point3f>&landmarks, std::vector<cv::Point2f>& feature_points) const ;

	//input start, inv_transform is the transformation matrix,  output is featurePoints-2D points of features, landmarks is 3D points of landmarkds
	void create_new_features(int start, const cv::Mat& inv_transform,  std::vector<cv::Point2f>& featurePoints, std::vector<cv::Point3f>& landmarks) const ;

	//this is to play on the image sequence
	void playSequence(const std::vector<std::vector<float>>& poses) const ;

	cv::Mat getK() const {return K;}
	float getBaseline() const {return baseline;}
	void setK(const cv::Mat& _K) {K = _K;}
	void setBundle(bool flag = false) {bundle = flag;}
	void setBaseline(float _baseline) {baseline = _baseline;}
	void set_Max_frame(int maxframe) {max_frame = maxframe;}
	void set_optimized_frame(int _optimizedframe) {optimized_frame = _optimizedframe;}

	//get the true pose files and convert it into vector form
	static std::vector<std::vector<float>> get_Pose(const std::string& path);

	//get the left image
	static cv::Mat getImage(const std::string& path, int i);

	//This is to remove duplicate points of the new frame feature points, the selected new feature points should be different from the first one
	static std::vector<int> removeDuplicate(const std::vector<cv::Point2f>& baseFeaturePoints, const std::vector<cv::Point2f>& newfeaturePoints, 
			const std::vector<int>& mask, int radius);

	//This is to track the new feature points from existing feature points
	//static std::vector<int> tracking(const cv::Mat& ref_img, const cv::Mat& curImg, const std::vector<cv::Point2f> featurePoints, 
	//			const std::vector<cv::Point3f>& landmarks, std::vector<cv::Point3f>& landmarks_ref, std::vector<cv::Point2f>&featurePoints_ref);
	static std::vector<int> tracking(const cv::Mat& ref_img, const cv::Mat& curImg, std::vector<cv::Point2f>& featurePoints, std::vector<cv::Point3f>& landmarks);

private:

	cv::Mat K;
	float baseline;
	int max_frame;
	int optimized_frame;
	std::string left_image_path;
	std::string right_image_path;


	// these to store all the 3d landmarks, history poses, and all frame informtion
	mutable std::vector<cv::Point3f> world_landmarks;
	mutable std::vector<cv::Point3f> history_pose;
	mutable std::vector<FrameInfo> framepoints;

	bool bundle;

	//This is to update the frame, start is the index of image to be retrieved, featurepoints is the feature points of new frame,
	//featurePoints_ref is the old featurePoints, tracked is the tracked feature points_ref, inlier is the inlier from PNP ransac, 
	// rvec and tvec is the pose estimation
	// void updateFrame(int start, const cv::Mat& inv_transform,std::vector<cv::Point2f>& featurePoints, const std::vector<cv::Point2f>& featurePoints_ref, 
	// 		const std::vector<int>&tracked, const std::vector<int>& inliers, const cv::Mat& rvec, const cv::Mat& tvec) const;

	std::vector<cv::Point2f> updateFrame(int i, const cv::Mat& inv_transform, const std::vector<cv::Point2f>& featurePoints, 
			const std::vector<int>&tracked, const std::vector<int>& inliers, const cv::Mat& rvec, const cv::Mat& tvec) const ;
	static inline double normofTransform( cv::Mat rvec, cv::Mat tvec ) {
   				 return fabs(std::min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
	}

};


#endif