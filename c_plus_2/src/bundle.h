#ifndef _BUNDEL_H
#define _BUNDEL_H

#include "vo.h"

//g2o
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
typedef g2o::LinearSolverEigen< SlamBlockSolver::PoseMatrixType > SlamLinearSolver;

// cvMat2Eigen, combine rvec, tvec to get the projection matrix,  rvec is euler angle rotation, tvec translation
Eigen::Isometry3d cvMat2Eigen( const cv::Mat& rvec, const cv::Mat& tvec );

cv::Mat Eigen2cvMat(const Eigen::Isometry3d& matrix);

g2o::SE3Quat euler2Quaternion( const cv::Mat& rvec, const cv::Mat& tvec );

//This one is not used for the time being
void BundleAdjust(const std::vector<FrameInfo>& frameinfo, const cv::Mat& K);

//return the last frame points pose information tvec
cv::Point3f BundleAdjust2(std::vector<FrameInfo>& frameinfo, std::vector<cv::Point3f>& world_points, 
		std::vector<cv::Point3f>& history_pose, const cv::Mat& K);

#endif
