import numpy as np
from numpy.linalg import inv, pinv
import cv2
from time import sleep

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
	#print(verts.shape, colors.shape)
	verts = np.hstack([verts.T, colors.T])
	with open(fn, 'wb') as f:
		f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
		np.savetxt(f, verts, fmt='%f %f %f %d %d %d')

def calculate_disparity(left_img, right_img):
		
		window_size = 11
		stereo = cv2.StereoSGBM_create(minDisparity = min_disp, 
										numDisparities=num_disp, 
										blockSize = 16,
										P1 = 8*3*window_size**2,
								        P2 = 32*3*window_size**2,
								        disp12MaxDiff = 1,
								        uniquenessRatio = 10,
								        speckleWindowSize = 100,
								        speckleRange = 32
								)
		disparity = stereo.compute(imgL,imgR).astype(np.float32) / 16.0
		return disparity

def  Transformation(p_C_points, intensities, f, frame_id):
	# Transform from camera frame to world frame
		R_C_frame = np.array([[0, -1, 0],[0, 0, -1], [1, 0, 0]])
		xlims = [7, 20]
		ylims = [-6, 10]
		zlims = [-5, 5]
		p_F_points = inv(R_C_frame).dot(p_C_points)

		mask = (
	    	(p_F_points[0, :] > xlims[0]) & (p_F_points[0, :] < xlims[1]) & 
	        (p_F_points[1, :] > ylims[0]) & (p_F_points[1, :] < ylims[1]) & 
	        (p_F_points[2, :] > zlims[0]) & (p_F_points[2, :] < zlims[1])
        )
		
		p_F_points = p_F_points[:,mask]
		intensities = intensities[mask]

		ss = f[frame_id].strip().split()
		T_W_C = [float(ele) for ele in ss]
		#print(T_W_C)
		T_W_C = np.array(T_W_C).reshape(3,4)

		#print(T_W_C)
		P_matrix = np.hstack((R_C_frame, np.zeros((3,1))))
		P_matrix = np.vstack((P_matrix, np.zeros((1,4))))
		P_matrix[3,3] = 1

		T_W_F = T_W_C.dot(P_matrix);

		points = T_W_F[:,0:3].dot(p_F_points) + T_W_F[:,3].reshape(3,1)

		return points, intensities

def disparityToPointCloud(disp_img, K, baseline, left_img):
		# points should be 3xN and intensities 1xN, where N is the amount of pixels
		# which have a valid disparity. I.e., only return points and intensities
		# for pixels of left_img which have a valid disparity estimate! The i-th
		# intensity should correspond to the i-th point.
		h, w = disp_img.shape

		h_ = np.linspace(1, h, h)
		w_ = np.linspace(1, w, w)
		[X,Y] = np.meshgrid(w_,h_);
		
		px_left =  np.stack((Y.reshape(-1), X.reshape(-1), np.ones((h*w))))

		disp_im = disp_img.reshape(-1)

		px_left = px_left[:, disp_im > 0]
		temp  = disp_im[disp_im > 0]

		# Switch from (row, col, 1) to (u, v, 1)
		px_left[0:2, :] = np.flipud(px_left[0:2, :])

		bv_left = inv(K).dot(px_left)

		f = K[0,0]

		x = f*baseline/temp
		points = bv_left*x

		intensities = left_img.reshape(-1)[disp_im > 0]

		return points, intensities


#change to your data path
baseline = 0.54
left_imgs = '../data/left/{0:06d}.png'
rhgt_imgs = '../data/right/{0:06d}.png'
poses 	  = '../data/poses.txt'

K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
			 [0, 7.188560000000e+02, 1.852157000000e+02],
			 [0, 0, 1]]
			 )
ground_truth = None
with open(poses) as f:
  ground_truth = f.readlines()

K[0:2, :] = K[0:2, :] / 2;

min_disp = 4
num_disp = 52 - min_disp
# print('computing disparity...')
# for i in range(0, maxNum):
# 	imgL = cv2.pyrDown(cv2.imread(left_imgs.format(i),0))
# 	imgR = cv2.pyrDown(cv2.imread(rhgt_imgs.format(i),0))
# 	stereo = cv2.StereoSGBM_create(minDisparity = min_disp, 
# 								numDisparities=num_disp, 
# 								blockSize=16,
# 								P1 = 8*3*window_size**2,
# 						        P2 = 32*3*window_size**2,
# 						        disp12MaxDiff = 1,
# 						        uniquenessRatio = 10,
# 						        speckleWindowSize = 100,
# 						        speckleRange = 32
# 								)
# 	#stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=11)
# 	#disparity = stereo.compute(imgL,imgR)
# 	disparity = stereo.compute(imgL,imgR).astype(np.float32) / 16.0
# 	# print(disparity)
# 	disp = (disparity - min_disp)/num_disp
# 	cv2.imshow('image', disp)
# 	sleep(0.01) # sleep 1.5 seconds
# 	k = cv2.waitKey(1) & 0xFF
# 	if k == 27:
# 		break

print('generating 3d point cloud...',)
out_fn = 'out.ply'

all_points = None
all_colors = None
maxNum = 100
for i in range(0, maxNum):

	imgL = cv2.resize(cv2.imread(left_imgs.format(i),0), (0,0), fx = 0.5, fy = 0.5)
	imgR = cv2.resize(cv2.imread(rhgt_imgs.format(i),0), (0,0), fx = 0.5, fy = 0.5)
	
	disparity = calculate_disparity(imgL, imgR)

	h, w = imgL.shape 

	Q = np.float32([[1, 0, 0, -303.5964],
                    [0, -1, 0,  92.6078], # turn points 180 deg around x-axis,
                    [0, 0, 0,  -359.428], # so that y-axis looks up
                    [0, 0, 1/baseline, 0]])

	#points = cv2.reprojectImageTo3D(disparity, Q).reshape(-1, 3)
	points, colors = disparityToPointCloud(disparity, K, baseline, imgL)
	points, colors = Transformation(points, colors, ground_truth, i)

	# with open('data.txt','wb') as f:
	# 	np.savetxt(f, points.T, fmt='%f %f %f')
	# colors  = imgL
	if len(imgL.shape) == 2:
		colors = np.dstack([colors]*3)
	if (len(colors.shape) == 3):
		colors = np.squeeze(colors, axis=0).T
	#colors = colors.reshape(3, -1)

	dis = disparity.reshape(-1)
	if all_points is None:
		all_points = points
		all_colors = colors
	else:
		all_points = np.hstack((all_points, points))
		all_colors = np.hstack((all_colors, colors))

	disp = (disparity - min_disp)/num_disp
	cv2.imshow('image', disp)
	sleep(0.01) # sleep 1.5 seconds
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break


write_ply(out_fn, all_points, all_colors)
cv2.destroyAllWindows()