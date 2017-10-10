import numpy as np
import cv2
from numpy.linalg import inv, pinv


def getK():
    return   np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
              [0, 7.188560000000e+02, 1.852157000000e+02],
              [0, 0, 1]])

def getKepoints():
    file = '../data/keypoints.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None)

def getTruePose():
    file = '/Users/HJK-BD//Downloads/kitti/poses/00.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None)

def getLandMarks():
    file = '../data/p_W_landmarks.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None)

def getLeftImage(i):
    return cv2.imread('/Users/HJK-BD//Downloads/kitti/00/image_0/{0:06d}.png'.format(i), 0)

def getRightImage(i):
    return cv2.imread('/Users/HJK-BD//Downloads/kitti/00/image_1/{0:06d}.png'.format(i), 0)    
  
def featureDetection(img, numCorners):

    h, w   = img.shape

    thresh = dict(threshold=24, nonmaxSuppression=True)
    fast   = cv2.FastFeatureDetector_create(**thresh)
    kp1    = fast.detect(img)
    kp1    = sorted(kp1, key = lambda x:x.response, reverse=True)[:numCorners]

    p1     = np.array([ele.pt for ele in kp1],dtype='int')
    # img3 = cv2.drawKeypoints(img, kp1, None, color=(255,0,0))
    # cv2.imshow('fast',img3)
    # cv2.waitKey(0) & 0xFF
    return p1

def featureTracking(img_1, img_2, p1, world_points):
    ##use KLT tracker
    lk_params = dict( winSize  = (21,21),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    ##find good one
    pre = p1[st==1]
    p2 = p2[st==1]
    w_points  = world_points[st==1]

    return w_points, pre,p2


def stereo_match_feature(left_img, right_img, patch_radius, keypoints, min_disp, max_disp):  
    # in case you want to find stereo match by yourself
    h, w = left_img.shape
    num_points = keypoints.shape[0]

    # Depth (or disparity) map
    depth = np.zeros(left_img.shape, np.uint8)
    output = np.zeros(keypoints.shape, dtype='int')
    all_index = np.zeros((keypoints.shape[0],1), dtype='int').reshape(-1)

    r     = patch_radius
    # patch_size = 2*patch_radius + 1;
      
    for i in range(num_points):

        row, col = keypoints[i,0], keypoints[i,1]
        # print(row, col)
        best_offset = 0;
        best_score = float('inf');

        if (row-r < 0 or row + r >= h or col - r < 0 or col + r >= w): continue

        left_patch = left_img[(row-r):(row+r+1), (col-r):(col+r+1)] # left imag patch    

        all_index[i] = 1

        for offset in range(min_disp, max_disp+1):

              if (row-r) < 0 or row + r >= h or  (col-r-offset) < 0 or (col+r-offset) >= w: continue
        
              diff  = left_patch - right_img[(row-r):(row+r+1), (col-r-offset):(col+r-offset+1)]
              sum_s = np.sum(diff**2)
 
              if sum_s < best_score:
                  best_score = sum_s
                  best_offset = offset

        output[i,0], output[i,1] = row,col-best_offset

    return output, all_index


def generate3D(featureL, featureR, K, baseline):
        # points should be 3xN and intensities 1xN, where N is the amount of pixels
        # which have a valid disparity. I.e., only return points and intensities
        # for pixels of left_img which have a valid disparity estimate! The i-th
        # intensity should correspond to the i-th point.

        temp = featureL - featureR
        temp = temp[:,1]

        print(featureL.shape, featureR.shape)
        
        px_left  =  np.vstack((featureL.T, np.ones((1, featureL.shape[0]))))
        # Switch from (row, col, 1) to (u, v, 1)
        px_left[0:2, :] = np.flipud(px_left[0:2, :])

        bv_left = inv(K).dot(px_left)

        f = K[0,0]

        z = f*baseline/temp
        points = bv_left*z

        #intensities = left_img.reshape(-1)[disp_im > 0]

        return points


def removeDuplicate(queryPoints, refPoints, radius=5):
    #remove duplicate points from new query points,
    for i in range(len(queryPoints)):
        query = queryPoints[i]
        xliml, xlimh = query[0]-radius, query[0]+radius
        yliml, ylimh = query[1]-radius, query[1]+radius
        inside_x_lim_mask = (refPoints[:,0] > xliml) & (refPoints[:,0] < xlimh)
        curr_kps_in_x_lim = refPoints[inside_x_lim_mask]

        if curr_kps_in_x_lim.shape[0] != 0:
            inside_y_lim_mask = (curr_kps_in_x_lim[:,1] > yliml) & (curr_kps_in_x_lim[:,1] < ylimh)
            curr_kps_in_x_lim_and_y_lim = curr_kps_in_x_lim[inside_y_lim_mask,:]
            if curr_kps_in_x_lim_and_y_lim.shape[0] != 0:
                queryPoints[i] =  np.array([0,0])
    return (queryPoints[:, 0]  != 0 )


def initiliazatize_3D_points(left_img, right_img, K, baseline):

    p1 = featureDetection(left_img, 500)

    p1 = np.fliplr(p1)

    #img_show  = cv2.imread('../data/left/{0:06d}.png'.format(0))

    p2, all_index = stereo_match_feature(left_img, right_img, 5, p1, 5, 50)

    p1 = p1[all_index > 0, :]
    p2 = p2[all_index > 0, :]

    M_left = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))

    M_rght = K.dot(np.hstack((np.eye(3), np.array([[-baseline,0, 0]]).T)))

    p1_flip = np.vstack((np.flipud(p1.T),np.ones((1,p1.shape[0]))))
    p2_flip    = np.vstack((np.flipud(p2.T),np.ones((1,p1.shape[0]))))

    # for p in p1:
    #     cv2.circle(img_show, (p[1], p[0]) ,1, (0,0,255), 2);

    P = cv2.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2]) 

    P = P/P[3]
    points = P[:3]

    # for p in p1:
    #     cv2.circle(left_img, (p[0], p[1]) ,1, (0,0,255), 2);

    # cv2.imshow('images', img_show)
    # k = cv2.waitKey(0) & 0xFF
    #print(points.T)
    return points.T, p1

def extract_keypoints_surf(img1, img2, K, baseline, refPoints = None):

    detector = cv2.xfeatures2d.SURF_create(400)
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict()   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc1,desc2,k=2)

    # ratio test as per Lowe's paper
    match_points1, match_points2 = [], []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            match_points1.append(kp1[m.queryIdx].pt)
            match_points2.append(kp2[m.trainIdx].pt)

    print('old lengthL', len(match_points1))

    p1 = np.array(match_points1).astype(float)
    p2 = np.array(match_points2).astype(float)

    if refPoints is not None:
        mask = removeDuplicate(p1, refPoints)
        p1 = p1[mask,:]
        p2 = p2[mask,:]

    print('new lengthL ', len(p1))

    M_left = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))

    M_rght = K.dot(np.hstack((np.eye(3), np.array([[-baseline,0, 0]]).T)))

    p1_flip = np.vstack((p1.T,np.ones((1,p1.shape[0]))))
    p2_flip = np.vstack((p2.T,np.ones((1,p2.shape[0]))))

    P = cv2.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2]) 

    P = P/P[3]
    land_points = P[:3]

    return land_points.T, p1

def playImageSequence(left_img, right_img, K):
    '''
        different ways to initialize the query points and landmark points
        you can specify the keypoints and landmarks
        or you can inilize_3D with FAST corner points, then stere match and then generate 3D points, but not so accurate
        or you can use the OPENCV feature extraction and matching functions
    '''

    #p1 = getKepoints().astype('float32')

    #print(p1)

    #points = getLandMarks()

    # points, p1 = initiliazatize_3D_points(left_img, right_img, K, baseline)
    # points = points.T
    #p1 = np.fliplr(p1).astype('float32')
    # print(points.shape)
    # print(p1.shape)

    points, p1 = extract_keypoints_surf(left_img, right_img, K, baseline)
    p1 = p1.astype('float32')

    pnp_objP = np.expand_dims(points, axis = 2)
    pnp_p1   = np.expand_dims(p1, axis = 2).astype(float)

    # reference
    reference_img = left_img
    reference_2D  = p1
    landmark_3D   = points

    #_, rvec, tvec = cv2.solvePnP(pnp_objP, pnp_p1, K, None)
    truePose = getTruePose()

    traj = np.zeros((600, 600, 3), dtype=np.uint8);

    maxError = 0

    for i in range(0, 2000):
            print('image: ', i)
            curImage =  getLeftImage(i)
           # curImage = cv2.imread('../data/left/{0:06d}.png'.format(i), 0)

            landmark_3D, reference_2D, tracked_2Dpoints = featureTracking(reference_img, curImage, reference_2D,  landmark_3D)

            # print(len(landmark_3D), len(valid_land_mark))

            pnp_objP = np.expand_dims(landmark_3D, axis = 2)
            pnp_cur  = np.expand_dims(tracked_2Dpoints, axis = 2).astype(float)

            _, rvec, tvec, inliers = cv2.solvePnPRansac(pnp_objP , pnp_cur, K, None)

            #update the new reference_2D
            reference_2D = tracked_2Dpoints[inliers[:,0],:]
            landmark_3D  = landmark_3D[inliers[:,0],:]
            
            #retrieve the rotation matrix
            rot,_ = cv2.Rodrigues(rvec)
            tvec = -rot.T.dot(tvec)     #coordinate transformation, from camera to world

            inv_transform = np.hstack((rot.T,tvec)) #inverse transform

            inliers_ratio = len(inliers)/len(pnp_objP) # the inlier ratio

            print('inliers ratio: ',inliers_ratio)

            # re-obtain the 3 D points if the conditions satisfied
            if (inliers_ratio < 0.9 or len(reference_2D) < 50):

                    ##initiliazation new landmarks
                    curImage_R = getRightImage(i)
                    # landmark_3D, reference_2D = initiliazatize_3D_points(curImage, curImage_R, K, baseline)
                    # reference_2D = np.fliplr(reference_2D).astype('float32')
                    landmark_3D_new, reference_2D_new  = extract_keypoints_surf(curImage, curImage_R, K, baseline, reference_2D)
                    reference_2D_new = reference_2D_new.astype('float32')
                    landmark_3D_new = inv_transform.dot(np.vstack((landmark_3D_new.T, np.ones((1,landmark_3D_new.shape[0])))))
                    valid_matches = landmark_3D_new[2,:] >0
                    landmark_3D_new = landmark_3D_new[:,valid_matches]

                    reference_2D = np.vstack((reference_2D, reference_2D_new[valid_matches,:]))
                    landmark_3D =  np.vstack((landmark_3D, landmark_3D_new.T))
        
            reference_img = curImage

            #draw images
            draw_x, draw_y = int(tvec[0]) + 300, int(tvec[2]) + 100;
            true_x, true_y = int(truePose[i][3]) + 300, int(truePose[i][11]) + 100

            curError = np.sqrt((tvec[0]-truePose[i][3])**2 + (tvec[1]-truePose[i][7])**2 + (tvec[2]-truePose[i][11])**2)
            print('Current Error: ', curError)
            if (curError > maxError):
                maxError = curError

            # print([truePose[i][3], truePose[i][7], truePose[i][11]])

            text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(tvec[0]), float(tvec[1]), float(tvec[2]));
            cv2.circle(traj, (draw_x, draw_y) ,1, (0,0,255), 2);
            cv2.circle(traj, (true_x, true_y) ,1, (255,0,0), 2);
            cv2.rectangle(traj, (10, 30), (550, 50), (0,0,0), cv2.FILLED);
            cv2.putText(traj, text, (10,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8);
            cv2.imshow( "Trajectory", traj );
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break

    #cv2.waitKey(0)
    print('Maximum Error: ', maxError)
    cv2.imwrite('map2.png', traj);
  #  imgpts, jac = cv2.projectPoints(pnp_objP, rvec, tvec, K, None)

    

if __name__ == '__main__':

    left_img    = getLeftImage(0)
    right_img   = getRightImage(0)

    baseline = 0.54;
    K =  getK()

    playImageSequence(left_img, right_img, K)


