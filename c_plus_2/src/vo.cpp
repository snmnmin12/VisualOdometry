#include "vo.h"
#include "bundle.h"

#include <fstream>
#include <string>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace cv;


VO::VO(const cv::Mat& _K, float _baseline, const string& _left_path, const string& _right_path):K(_K), baseline(_baseline){

	left_image_path = _left_path;
	right_image_path = _right_path;

}

vector<Point3f> VO::get3D_Points(const vector<Point2f>& feature_p1, const vector<Point2f>& feature_p2) const {

	// This is to generate the 3D points

	Mat M_left  = Mat::zeros(3,4, CV_64F);
    Mat M_right = Mat::zeros(3,4, CV_64F);
    M_left.at<double>(0,0) =1; M_left.at<double>(1,1) =1;M_left.at<double>(2,2) =1;
    M_right.at<double>(0,0) =1; M_right.at<double>(1,1) =1;M_right.at<double>(2,2) =1;
    M_right.at<double>(0,3) =-baseline;

    M_left  = K*M_left;
    M_right = K*M_right;

    Mat landmarks;
    cout << "3D points: "<<feature_p1.size() << ": " << feature_p2.size() << endl;
    triangulatePoints(M_left, M_right, feature_p1, feature_p2, landmarks); 

    // cout << landmarks.type()<<endl;
    // cout << landmarks.size() << endl;

    std::vector<Point3f> output;

    for (int i = 0; i < landmarks.cols; i++) {
    	Point3f p;
    	p.x = landmarks.at<float>(0, i)/landmarks.at<float>(3, i);
		p.y = landmarks.at<float>(1, i)/landmarks.at<float>(3, i);
		p.z = landmarks.at<float>(2, i)/landmarks.at<float>(3, i);
		output.push_back(p);
    }

    return output;
}

void VO::extract_keypoints_surf(const Mat& img1, const Mat& img2, vector<Point3f>&landmarks, vector<Point2f>& feature_points) const {

	// Feature Detection and Extraction
	Ptr<FeatureDetector> detector = ORB::create(352);

	vector<KeyPoint> keypoints1, keypoints2;
	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);

	// // cv::SurfDescriptorExtractor extractor;
	// cv::Ptr<cv::SURF> extractor = cv::SURF::create();
	Ptr<DescriptorExtractor> extractor = ORB::create();
	
	cv::Mat descriptors1, descriptors2;
	extractor->compute(img1, keypoints1, descriptors1);
	extractor->compute(img2, keypoints2, descriptors2);

	descriptors1.convertTo(descriptors1, CV_32F);
    descriptors2.convertTo(descriptors2, CV_32F);
	// cv::BruteForceMatcher<L2<float> > matcher;

	FlannBasedMatcher matcher;
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descriptors1, descriptors2, matches, 2);

    vector<Point2f> match_points1;
    vector<Point2f> match_points2;

    // std::vector<DMatch> bestMatches;

    // cout << matches.size() << endl;

    for (int i = 0; i < matches.size(); i++) {

    	const DMatch& bestMatch = matches[i][0];  
		const DMatch& betterMatch = matches[i][1];  

		if (bestMatch.distance < 0.7*betterMatch.distance) {
			match_points1.push_back(keypoints1[bestMatch.queryIdx].pt);
            match_points2.push_back(keypoints2[bestMatch.trainIdx].pt);
            // bestMatches.push_back(bestMatch);
		}

    }

   feature_points = match_points1;

   landmarks = get3D_Points(match_points1, match_points2);

}


void VO::create_new_features(int start, const Mat& inv_transform, std::vector<Point2f>& featurePoints, std::vector<Point3f>& landmarks)  const{

	if (featurePoints.size() != 0) {
		featurePoints.clear();
		landmarks.clear();
	}

	Mat curImage_L = getImage(left_image_path, start);
	Mat curImage_R = getImage(right_image_path, start);

	vector<Point3f>  landmark_3D_new;
	vector<Point2f>  reference_2D_new;

	extract_keypoints_surf(curImage_L, curImage_R, landmark_3D_new, reference_2D_new);

	// // cout << inv_transform << endl;

   for (int k = 0; k < landmark_3D_new.size(); k++) {
// 
	   	const Point3f& pt = landmark_3D_new[k];

	   	Point3f p;

	   	p.x = inv_transform.at<double>(0, 0)*pt.x + inv_transform.at<double>(0, 1)*pt.y + inv_transform.at<double>(0, 2)*pt.z + inv_transform.at<double>(0, 3);
	   	p.y = inv_transform.at<double>(1, 0)*pt.x + inv_transform.at<double>(1, 1)*pt.y + inv_transform.at<double>(1, 2)*pt.z + inv_transform.at<double>(1, 3);
	   	p.z = inv_transform.at<double>(2, 0)*pt.x + inv_transform.at<double>(2, 1)*pt.y + inv_transform.at<double>(2, 2)*pt.z + inv_transform.at<double>(2, 3);

	   	// cout << p << endl;
	   	if (p.z > 0) {
			landmarks.push_back(p);
			featurePoints.push_back(reference_2D_new[k]);
	   	}

   }

}

// vector<int> VO::tracking(const cv::Mat& ref_img, const cv::Mat& curImg, const std::vector<Point2f> featurePoints, const std::vector<Point3f>& landmarks, 
// 	std::vector<Point3f>&  landmarks_ref, std::vector<Point2f>&featurePoints_ref) {
vector<int> VO::tracking(const cv::Mat& ref_img, const cv::Mat& curImg, std::vector<Point2f>& featurePoints, std::vector<Point3f>& landmarks) {

		vector<Point2f> nextPts;
		vector<uchar> status;
		vector<float> err;

		calcOpticalFlowPyrLK(ref_img, curImg, featurePoints, nextPts, status, err);

		std::vector<int> res;
		featurePoints.clear();
		// cout << featurePoints.size() << ", " << landmarks.size() << ", " << status.size() << endl;

		for (int  j = status.size()-1; j > -1; j--) {
			if (status[j] != 1) {
				// featurePoints.erase(featurePoints.begin()+j);
				landmarks.erase(landmarks.begin()+j);
		
			} else {
				featurePoints.push_back(nextPts[j]);
				res.push_back(j);
			}
		}
		std::reverse(res.begin(),res.end()); 
		std::reverse(featurePoints.begin(),featurePoints.end()); 

		return res;
}

vector<int> VO::removeDuplicate(const vector<Point2f>& baseFeaturePoints, const vector<Point2f>& newfeaturePoints, 
	const vector<int>& mask, int radius=10)
{	

	std::vector<int> res;
	for (int j = 0; j < newfeaturePoints.size(); j++) {

		const Point2f& p2 = newfeaturePoints[j];

		bool within_range = false;

		for (auto index : mask) {
			const Point2f& p1 = baseFeaturePoints[index];
			if (cv::norm(p1-p2) < radius) {
				within_range = true;
				break;
			}
		}
		if (!within_range) res.push_back(j);
	}

	return res;

}

//void VO::updateFrame(int i, const cv::Mat& inv_transform, vector<Point2f>& featurePoints, const vector<Point2f>& featurePoints_ref, 
//			const vector<int>&tracked, const vector<int>& inliers, const Mat& rvec, const Mat& tvec) const {

vector<Point2f> VO::updateFrame(int i, const cv::Mat& inv_transform, const vector<Point2f>& featurePoints, 
			const vector<int>&tracked, const vector<int>& inliers, const Mat& rvec, const Mat& tvec) const {

			vector<Point2f> new_2D;
			vector<Point3f> new_3D;

			create_new_features(i, inv_transform, new_2D, new_3D);

			// featurePoints.clear();
			std::vector<Point2f> up_featurePoints;
			// landmarks.clear();

			const std::vector<int>& preIndexes = framepoints.back().obj_indexes;
			// for (int j = 0; j < preIndexes.size(); j++)
			// 	cout << preIndexes[j] << " ";
			// cout << endl;

			// cout << featurePoints_ref.size() << ": " << new_2D.size() << ": " << inliers.size() << endl;
			vector<int> res = removeDuplicate(featurePoints, new_2D, inliers);
			cout << res.size() << ": " << new_2D.size() << endl;

			// if (res.size() < 2) return up_featurePoints;

				std::vector<int> indexes;
				// cout << "tracked size: " << tracked.size() << "inlier size" << inliers.size() << endl;

				for (auto index : inliers) {
					// cout << index << " ";
					up_featurePoints.push_back(featurePoints[index]);
					//featurePoints.push_back(featurePoints_ref[index]);
					// landmarks.push_back(landmarks_ref[index]);
					indexes.push_back(preIndexes[tracked[index]]);
				}
				// cout << endl;

				int start = world_landmarks.size();

				for (auto index : res) {
					up_featurePoints.push_back(new_2D[index]);
					// landmarks.push_back(new_3D[index]);
					world_landmarks.push_back(new_3D[index]);
					indexes.push_back(start++);
				}

				///for check correctness 
				// for (int k = 0; k < landmarks.size(); k++) {
				// 	if (landmarks[k] != world_landmarks[indexes[k]])
				// 	throw std::invalid_argument("These two landmarks should be the same!");
				// }

				FrameInfo frameinfo;
				frameinfo.features2d = up_featurePoints;
				frameinfo.obj_indexes = indexes;
				// frameinfo.features3d = landmarks;
				frameinfo.rvec  = rvec;
				frameinfo.tvec  = tvec;
				framepoints.push_back(frameinfo);
				if (framepoints.size() > optimized_frame) {
					framepoints.erase(framepoints.begin());
				}

				return up_featurePoints;

}

void VO::playSequence(const std::vector<std::vector<float>>& poses) const {

	int startIndex = 0;

	vector<Point3f> location_history;

	Mat left_img = getImage(left_image_path, startIndex);
	Mat right_img = getImage(right_image_path,startIndex);
	Mat& ref_img  = left_img;

	// vector<Point3f> landmarks;
	vector<Point2f> featurePoints;
	//vector<int> landmarks;

	bool visualize = true;
	//bool bundle = true;
	// int optimized_frame = ;

	//extract_keypoints_surf(left_img, right_img, landmarks, featurePoints);
	extract_keypoints_surf(left_img, right_img, world_landmarks, featurePoints);

	// cout << featurePoints[0] << endl;
	// cout << landmarks[0] << endl;

	// vector<FrameInfo> framepoints;
	//initilize the first frame with the below parameters
	{

		FrameInfo frameinfo;
		frameinfo.features2d = featurePoints;
		for (int i = 0; i < featurePoints.size(); i++) frameinfo.obj_indexes.push_back(i);
		// frameinfo.features3d = landmarks;
		// world_landmarks = landmarks;
		frameinfo.rvec  = Mat::zeros(3,1,CV_64F);
		frameinfo.tvec  = Mat::zeros(3,1,CV_64F);
		framepoints.push_back(frameinfo);
	}
	int start = 0;

	Mat curImg;

	Mat traj = Mat::zeros(600, 600, CV_8UC3);

	for(int i = startIndex + 1; i < max_frame; i++) {

		cout << i << endl;

		curImg = getImage(left_image_path, i);

		//std::vector<Point3f>  landmarks_ref, landmarks;
		//std::vector<Point2f>  featurePoints_ref;
		std::vector<Point3f>  landmarks;

		featurePoints = framepoints.back().features2d;

		for (auto index : framepoints.back().obj_indexes) {
			landmarks.push_back(world_landmarks[index]);
		}

		vector<int> tracked = tracking(ref_img, curImg, featurePoints, landmarks);// landmarks_ref, featurePoints_ref);
		// cout << featurePoints.size() << ", " << landmarks.size() << endl;
		if (landmarks.size() < 10) continue;

		Mat dist_coeffs = Mat::zeros(4,1,CV_64F);

		Mat rvec, tvec;
		
		vector<int> inliers;
		// cout << featurePoints.size() << ", " << landmarks.size() << endl;
		solvePnPRansac(landmarks, featurePoints, K, dist_coeffs,rvec, tvec,false, 100, 8.0, 0.99, inliers);// inliers);

		if (inliers.size() < 5) continue;

		// cout << "Norm: " << normofTransform(rvec- framepoints.back().rvec, tvec - framepoints.back().tvec)  << endl;
		// if ( normofTransform(rvec- framepoints.back().rvec, tvec - framepoints.back().tvec) > 2 ) continue;

		// if (normofTransform(rvec, tvec) > 0.3) continue;

		float inliers_ratio = inliers.size()/float(landmarks.size());

		cout << "inliers ratio: " << inliers_ratio << endl;


		Mat R_matrix;
		Rodrigues(rvec,R_matrix); 
		R_matrix = R_matrix.t();
		Mat t_vec = -R_matrix*tvec;

		// cout << t_vec << endl;
		Mat inv_transform = Mat::zeros(3,4,CV_64F);
		R_matrix.col(0).copyTo(inv_transform.col(0));
		R_matrix.col(1).copyTo(inv_transform.col(1));
		R_matrix.col(2).copyTo(inv_transform.col(2));
		t_vec.copyTo(inv_transform.col(3));


		featurePoints = updateFrame(i, inv_transform, featurePoints, tracked, inliers, rvec, tvec);

		if (featurePoints.size() == 0) continue;

		//featurePoints = up_featurePoints;
		ref_img = curImg;

		t_vec.convertTo(t_vec, CV_32F);

		if (bundle  && (framepoints.size() == optimized_frame || i == max_frame - 1)) {
				Point3f p3 = BundleAdjust2(framepoints, world_landmarks, location_history, K);
				framepoints.erase(framepoints.begin()+1, framepoints.end()-1);
				history_pose.push_back(p3);
		} else {
			history_pose.push_back(Point3f(t_vec.at<float>(0), t_vec.at<float>(1), t_vec.at<float>(2)));
		}
		cout << t_vec.t() << endl;
		cout << "truth" << endl;
		cout << "["<<poses[i][3] << ", " << poses[i][7] << ", " << poses[i][11] <<"]"<<endl;

		if (visualize) {
					// plot the information
			string text  = "Red color: estimated trajectory";
			string text2 = "Blue color: Groundtruth trajectory";
			// cout << framepoints.size() << endl; 
			// cout << t_vec.t() << endl;
			// cout << "["<<poses[i][3] << ", " << poses[i][7] << ", " << poses[i][11] <<"]"<<endl;
			Point2f center  = Point2f(int(t_vec.at<float>(0)) + 300, int(t_vec.at<float>(2)) + 100);
			Point2f center2 = Point2f(int(poses[i][3]) + 300, int(poses[i][11]) + 100);

			circle(traj, center , 1, Scalar(0,0,255), 1);
			circle(traj, center2, 1, Scalar(255,0,0), 1);
			rectangle(traj, Point2f(10, 30), Point2f(550, 50),  Scalar(0,0,0), cv::FILLED);
			putText(traj, text,  Point2f(10,50), cv::FONT_HERSHEY_PLAIN, 1, Scalar(0, 0,255), 1, 5);
			putText(traj, text2, Point2f(10,70), cv::FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0), 1, 5);

			if (bundle) {
					string text1 = "Green color: bundle adjusted trajectory";
					for (const auto& p3 : location_history) {
						Point2f center1 = Point2f(int(p3.x) + 300, int(p3.z) + 100);
						circle(traj, center1, 1, Scalar(0,255,0), 1);
					}
					location_history.clear();
					putText(traj, text1, Point2f(10,90), cv::FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0), 1, 5);
			}

			imshow( "Trajectory", traj);
			waitKey(1);
		}

	}

	// if (bundle) {

	// 	// BundleAdjust2(framepoints, world_landmarks, history_pose, K);
	// }

	if (visualize) {

		// if (bundle) {
		// 	string text1 = "Green color: bundle adjusted trajectory";
		// 	putText(traj, text1, Point2f(10,90), cv::FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0), 1, 5);
		// 	for (const auto& p3 : history_pose) {
		// 		Point2f center1 = Point2f(int(p3.x) + 300, int(p3.z) + 100);
		// 		circle(traj, center1, 1, Scalar(0,255,0), 1);
		// 	}
		// 	imshow( "Trajectory", traj);
		// }
		imwrite("map2.png", traj);
		waitKey(0);
	}

}


//############################################################
// static methods here

std::vector<vector<float>> VO::get_Pose(const std::string& path) {

  std::vector<vector<float>> poses;
  // ifstream myfile("/Users/HJK-BD/Downloads/kitti/poses/00.txt");
  ifstream myfile(path);
  string line;
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
          char * dup = strdup(line.c_str());
   		  char * token = strtok(dup, " ");
   		  std::vector<float> v;
	   	  while(token != NULL){
	        	v.push_back(atof(token));
	        	token = strtok(NULL, " ");
	    	}
	    	poses.push_back(v);
	    	free(dup);
    }
    myfile.close();
  } else {
  	cout << "Unable to open file"; 
  }	

  return poses;

}



Mat VO::getImage(const string& raw_path, int i) {
	char path[50];
	// sprintf(path, "/Users/HJK-BD//Downloads/kitti/00/image_0/%06d.png", i);
	sprintf(path, raw_path.c_str(), i);
	return imread(path);
}
