
#ifndef VO_H
#define VO_H


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"


class VO {

public:

	VO(){}
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
	void setBaseline(float _baseline) {baseline = _baseline;}
	void set_Max_frame(int maxframe) {max_frame = maxframe;}

	//get the true pose files and convert it into vector form
	static std::vector<std::vector<float>> get_Pose(const std::string& path);
	//get the left image
	static cv::Mat getImage(const std::string& path, int i);

private:
	cv::Mat K;
	float baseline;
	int max_frame;
	std::string left_image_path;
	std::string right_image_path;
};


#endif