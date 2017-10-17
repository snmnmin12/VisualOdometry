
#include "vo.h"
using namespace std;

int main(int argc, char** argv) {


	if (argc != 2)
    {
        cout<<"Usage: ba max_frame"<<endl;
        exit(1);
    }

    int max_frame= atoi(argv[1]);

	//you have to configure your own path
	string left_path =  "/Users/HJK-BD/Downloads/kitti/00/image_0/%06d.png";
	string right_path = "/Users/HJK-BD/Downloads/kitti/00/image_1/%06d.png";
	string pose_path = "/Users/HJK-BD/Downloads/kitti/poses/00.txt";

	cout <<"Program starts!"<<endl;

	cv::Mat K = (cv::Mat_<double>(3, 3) << 7.188560000000e+02, 0, 6.071928000000e+02,
                         0, 7.188560000000e+02, 1.852157000000e+02,
                         0, 0, 1);

	float baseline = 0.54;

	VO vo(K, baseline, left_path, right_path);

	//we have only 1000 paris of pictures
	vo.set_Max_frame(max_frame);

	//This is to get the ground truethe pose data
	vector<vector<float>> poses = VO::get_Pose(pose_path);

	vo.playSequence(poses);

	cout << "You come to the end!" << endl;

	return 0;
}
