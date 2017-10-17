

#include "bundle.h"
#include <unordered_map>
#include <unordered_set>

// typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
// typedef g2o::LinearSolverEigen< SlamBlockSolver::PoseMatrixType > SlamLinearSolver;

// cvMat2Eigen
cv::Mat Merge( const cv::Mat& rvec, const cv::Mat& tvec ){

	cv::Mat R;
    cv::Rodrigues( rvec, R );
    cv::Mat T = cv::Mat::zeros(3,4, CV_64F);
    T.at<double>(0,0) = R.at<double>(0,0);
    T.at<double>(0,1) = R.at<double>(0,1);
    T.at<double>(0,2) = R.at<double>(0,2);
    T.at<double>(1,0) = R.at<double>(1,0);
    T.at<double>(1,1) = R.at<double>(1,1);
    T.at<double>(1,2) = R.at<double>(1,2);
    T.at<double>(2,0) = R.at<double>(2,0);
    T.at<double>(2,1) = R.at<double>(2,1);
    T.at<double>(2,2) = R.at<double>(2,2);

    T.at<double>(0,3) = tvec.at<double>(0);
    T.at<double>(1,3) = tvec.at<double>(1);
    T.at<double>(2,3) = tvec.at<double>(2);

    // cv::hconcat(R, tvec, T);
    // std::cout << T << std::endl;
    return T;

}

// cvMat2Eigen
Eigen::Isometry3d cvMat2Eigen( const cv::Mat& rvec, const cv::Mat& tvec ){

	cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ ) 
            r(i,j) = R.at<double>(i,j);
  
    // 将平移向量和旋转矩阵转换成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = tvec.at<double>(0,0); 
    T(1,3) = tvec.at<double>(1,0); 
    T(2,3) = tvec.at<double>(2,0);

    return T;

}

//将eigen 矩阵转换为opencv矩阵
cv::Mat Eigen2cvMat(const Eigen::Isometry3d& matrix) {

	cv::Mat R = cv::Mat::zeros(3,3,CV_64F);
	cv::Mat tvec = cv::Mat::zeros(1,3,CV_64F);

    for ( int i=0; i<3; i++ )
    for ( int j=0; j<3; j++ ) 
        R.at<double>(i,j) = matrix(i,j);

    tvec.at<double>(0) = matrix(0, 3); 
    tvec.at<double>(1) = matrix(1, 3);  
    tvec.at<double>(2) = matrix(2, 3);

    tvec = -R.t()*tvec.t(); 

    return tvec;

}

//欧拉 旋转 转换为 四元角 表达
g2o::SE3Quat euler2Quaternion(const cv::Mat& rvec, const cv::Mat& tvec )
{

	// std::cout << rvec.size() << std::endl;
	// std::cout << tvec.size() << std::endl;

	cv::Mat R;
    cv::Rodrigues( rvec, R );

	double roll = rvec.at<double>(0,0);
	double pitch = rvec.at<double>(1,0);
	double yaw = rvec.at<double>(2,0);

    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd yawAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = rollAngle * yawAngle * pitchAngle;

    Eigen::Vector3d trans(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    g2o::SE3Quat pose(q,trans);
    // std::cout << pose << std::endl;
    // std::cout << trans << std::endl;

    // assert(false);

    return pose;
}

void print(const std::vector<FrameInfo>& frameinfo, g2o::SparseOptimizer& optimizer) {


	for ( size_t i=0; i < frameinfo.size(); i++ )
    {

    	// if ( i < frameinfo.size() - 1) continue;

        g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(i));
        // std::cout<<"vertex id "<<i<<", pos = " << std::endl;
        Eigen::Isometry3d pose = v->estimate();
        cv::Mat tvec = Eigen2cvMat(pose);
        // std::cout<<pose.matrix()<<std::endl;
        // g2o::SE3Quat pose2 = euler2Quaternion(frameinfo[i].rvec, frameinfo[i].tvec);
        // std::cout<<pose2<<std::endl;
        std::cout << "optimized: " << std::endl << tvec << std::endl;

       	cv::Mat R;
    	cv::Rodrigues( frameinfo[i].rvec, R );
        std::cout<< "original: "<<std::endl << -R.t()*frameinfo[i].tvec <<std::endl;
    }
}



void BundleAdjust(const std::vector<FrameInfo>& frameinfo, const cv::Mat& K) {

    // 构造g2o中的图
    // 先构造求解器
// 	g2o::SparseOptimizer    optimizer;
//     auto linearSolver = g2o::make_unique<SlamLinearSolver>();
//     linearSolver->setBlockOrdering( false );
//     // L-M 下降 
//     g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<SlamBlockSolver>(std::move(linearSolver)));
    
//     optimizer.setAlgorithm( algorithm );
//     optimizer.setVerbose( false );


//     //camera information
//     g2o::CameraParameters* camera = new g2o::CameraParameters(K.at<double>(0,0), Eigen::Vector2d(K.at<double>(0,2), K.at<double>(1,2)), 0 );
//     camera->setId(0);
//     optimizer.addParameter( camera);

//     int size = frameinfo.size();

// //add vertext
//     for ( int i=0; i< size; i++ )
//     {
//         g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
//         v->setId(i);
//         if ( i == 0) v->setFixed( true );
//         // Eigen::Isometry3d T = cvMat2Eigen(frameinfo[i].rvec, frameinfo[i].tvec);
//         g2o::SE3Quat pose = euler2Quaternion(frameinfo[i].rvec, frameinfo[i].tvec);
//         v->setEstimate( pose );
//         optimizer.addVertex( v );
//     }

//     int startIndex = size;

//     std::cout <<"start Index: " << startIndex << std::endl;

//     for ( int i=0; i< size; i++ )
//     {

//     	for (int j = 0; j < frameinfo[i].features3d.size(); j++) {

//     		//add the landmark
// 	        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
// 	        v->setId(startIndex);
// 	        v->setMarginalized(true);
// 	        v->setEstimate(Eigen::Vector3d(frameinfo[i].features3d[j].x, frameinfo[i].features3d[j].y, frameinfo[i].features3d[j].z));
// 	        optimizer.addVertex( v );

// 	        //add the edges
// 	         g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
// 	         edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(startIndex)));
// 	         edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>   (optimizer.vertex(i)) );

// 	         edge->setMeasurement( Eigen::Vector2d(frameinfo[i].features2d[j].x, frameinfo[i].features2d[j].y));
// 	         edge->setInformation( Eigen::Matrix2d::Identity() );
// 	         edge->setParameterId(0, 0);
// 	         // 核函数
// 	         edge->setRobustKernel( new g2o::RobustKernelHuber() );
// 	         optimizer.addEdge( edge );
// 	         startIndex++;
//     	}
//     }

//     std::cout<<"optimizing pose graph, vertices: "<<optimizer.vertices().size()<<std::endl;
//     optimizer.initializeOptimization();

//     optimizer.optimize( 100 ); //可以指定优化步数

//     // 以及所有特征点的位置
//     print(frameinfo, optimizer);

//     optimizer.clear();
//     std::cout<<"Optimization done."<<std::endl;

}


cv::Point3f BundleAdjust2(std::vector<FrameInfo>& frameinfo, std::vector<cv::Point3f>& world_points, std::vector<cv::Point3f>& history_poses, const cv::Mat& K) {

	// g2o::SparseOptimizer    optimizer;
 //    auto linearSolver = g2o::make_unique<SlamLinearSolver>();
 //    linearSolver->setBlockOrdering( false );
 //    // L-M 下降 
 //    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<SlamBlockSolver>(std::move(linearSolver)));
    
 //    optimizer.setAlgorithm( algorithm );
 //    optimizer.setVerbose( false );
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );

    g2o::SparseOptimizer optimizer;  // 最后用的就是这个东东
    optimizer.setAlgorithm( solver ); 
    // 不要输出调试信息
    optimizer.setVerbose( false );


    //camera information
    g2o::CameraParameters* camera = new g2o::CameraParameters(K.at<double>(0,0), Eigen::Vector2d(K.at<double>(0,2), K.at<double>(1,2)), 0 );
    camera->setId(0);
    optimizer.addParameter( camera);

    int size = frameinfo.size();

	//add vertext
    for ( int i=0; i< size; i++ )
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if ( i == 0) v->setFixed( true );
        // Eigen::Isometry3d T = cvMat2Eigen(frameinfo[i].rvec, frameinfo[i].tvec);
        g2o::SE3Quat pose = euler2Quaternion(frameinfo[i].rvec, frameinfo[i].tvec);
        v->setEstimate( pose );
        optimizer.addVertex( v );
    }

    int startIndex = size;

    std::cout <<"start Index: " << startIndex << std::endl;

    std::unordered_map<int, int> has_seen;
    int count = 0;

    for (int i = 0; i < size; i++) {

    	const FrameInfo& frame = frameinfo[i];

    	for (int j = 0; j < frame.obj_indexes.size(); j++) {	

    			int index = frame.obj_indexes[j];
	    		int currentNodeIndex = startIndex;

	    	if (has_seen.find(index) == has_seen.end()) {	
	    			//add the landmark
		        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
		        v->setId(startIndex);
		        v->setMarginalized(true);
		        cv::Point3f w_point = world_points[index];
		        v->setEstimate(Eigen::Vector3d(w_point.x, w_point.y, w_point.z));
		        optimizer.addVertex( v );
		        has_seen[index] = startIndex;
		        startIndex++;
	    	}  else {

	    		currentNodeIndex = has_seen[index];
	    	}

	        //add the edges
	         g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
	         edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(currentNodeIndex)));
	         edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>   (optimizer.vertex(i)));
	         edge->setMeasurement( Eigen::Vector2d(frame.features2d[j].x, frame.features2d[j].y));
	         edge->setInformation( Eigen::Matrix2d::Identity() );
	         edge->setParameterId(0, 0);
	         edge->setRobustKernel( new g2o::RobustKernelHuber() );
	         optimizer.addEdge( edge );
    	}
    }
   
    std::cout<<"optimizing pose graph, vertices: "<<optimizer.vertices().size()<<std::endl;
    optimizer.initializeOptimization();

    optimizer.optimize(10); //可以指定优化步数

    // 以及所有特征点的位置
    // print(frameinfo, optimizer);
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(size-1));
    Eigen::Isometry3d pose = v->estimate();
    cv::Mat tvec = Eigen2cvMat(pose);
    // std::cout << "tvec is : " << tvec.at<double>(0) << ", " << tvec.at<double>(1) << ", " << tvec.at<double>(2) << std::endl;
    // history_poses.push_back(cv::Point3f(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)));
    {

                            //         g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(i));       
                //         Eigen::Isometry3d pose = v->estimate();
                //         cv::Mat tvec = Eigen2cvMat(pose);

    


                for ( size_t i=2; i < frameinfo.size(); i++ )
                    {
                        g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(i));       
                        Eigen::Isometry3d pose = v->estimate();
                        cv::Mat t_vec = Eigen2cvMat(pose);
                        //cv::Mat R;
                        //cv::Rodrigues( frameinfo[i].rvec, R );
                        //cv::Mat t_vec =  -R.t()*frameinfo[i].tvec;
                        history_poses.push_back(cv::Point3f(t_vec.at<double>(0), t_vec.at<double>(1), t_vec.at<double>(2)));
                    }

    	// for (auto& item : has_seen) {
    	// 	g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(item.second));
    	// 	Eigen::Vector3d pos = v->estimate();
    	// 	world_points[item.first] = cv::Point3f(pos[0], pos[1], pos[2]);
    	// }

    }
    optimizer.clear();
    std::cout<<"Optimization done."<<std::endl;

    return cv::Point3f(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
}