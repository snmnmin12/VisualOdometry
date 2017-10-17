
#include "vo.h"
using namespace std;

int main(int argc, char** argv) {

	 if (argc != 4)
    {
        cout<<"Usage: ba_example max_frame, bundle"<<endl;
        exit(1);
    } 

    int max_frame= atoi(argv[1]);
    int opt_frame = atoi(argv[2]);
    bool flag = atoi(argv[3]);

    // std::cout << flag << std::endl;

	//you have to configure your own path
	string left_path =  "/Users/HJK-BD/Downloads/kitti/00/image_0/%06d.png";
	string right_path = "/Users/HJK-BD/Downloads/kitti/00/image_1/%06d.png";
	string pose_path = "/Users/HJK-BD/Downloads/kitti/poses/00.txt";

	cv::Mat K = (cv::Mat_<double>(3, 3) << 7.188560000000e+02, 0, 6.071928000000e+02,
                         0, 7.188560000000e+02, 1.852157000000e+02,
                         0, 0, 1);

	float baseline = 0.54;
	
	VO vo(K, baseline, left_path, right_path);

	//we have only 1000 paris of pictures
	vo.set_Max_frame(max_frame);
	vo.set_optimized_frame(opt_frame);
	vo.setBundle(flag);

	//This is to get the ground truethe pose data
	vector<vector<float>> poses = VO::get_Pose(pose_path);

	vo.playSequence(poses);

	cout << "You come to the end!" << endl;

	return 0;
}