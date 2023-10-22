import os
import numpy as np
import cv2
from scipy.optimize import least_squares
import time
import random
from videostream_D455 import *
from scale_factor import *


from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

from matplotlib import pyplot as plt

import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.rotations as pr
import pytransform3d .camera as pc

from cycler import cycle

start_time = time.time()

class VisualOdometry():
    def __init__(self, data_dir):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib('stereocalib.txt')
        
        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)

        

        self.fastFeatures = cv2.FastFeatureDetector_create()

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """
        with open(filepath, 'r') as f:
            line = f.readline().strip()
            params = np.fromstring(line, dtype=np.float32, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            P_l[0, 3], P_l[1, 3] = params[3], params[7] 
           
            line = f.readline().strip()
            params = np.fromstring(line, dtype=np.float32, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
            P_r[0, 3], P_r[1, 3] = params[3], params[7] 

        return K_l, P_l ,K_r, P_r

    @staticmethod
   
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images. Shape (n, height, width)
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        return images

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # if self.PrintTimes <10 :
        #    print(f"rotation vector {self.PrintTimes+1}: {R}")
        
        print()
        # Get the translation vector
        t = dof[3:]

        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def get_tiled_keypoints(self, img, tile_h, tile_w):

        """AndCompute
        Splits the image into tiles and detects the 10 best keypoints in each tile

        Parameters
        ----------
        img (ndarray): The image to find keypoints in. Shape (height, width)
        tile_h (int): The tile height
        tile_w (int): The tile width

        Returns
        -------
        kp_list (ndarray): A 1-D list of all keypoints. Shape (n_keypoints)
        """
        def get_kps(x, y):
            # Get the image tile
            impatch = img[y:y + tile_h, x:x + tile_w]

            # Detect keypoints
            keypoints = self.fastFeatures.detect(impatch)

            # Sort keypoints based on response
            keypoints = sorted(keypoints, key=lambda k: -k.response)

            # Get the 10 best keypoints
            keypoints = keypoints[:10]

            # Correct the coordinate for each keypoint
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            return keypoints

        # Get the image height and width
        h, w, *_ = img.shape

        # Get the keypoints for each of the tiles
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        # Flatten the keypoint list
        kp_list_flatten = np.concatenate(kp_list)

        return kp_list_flatten
       
 
    def track_keypoints(self, img1, img2, kp1, max_error=4):
        """
        Tracks the keypoints between frames

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error

        Returns
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """


       # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a mask that selects the keypoints that were trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints*
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints that are outside the image
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints11 = trackpoints1[in_bounds]
        trackpoints22 = trackpoints2[in_bounds]  

        return trackpoints11, trackpoints22

        

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=64):

        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        in_bounds = np.logical_and(mask1, mask2)
        
        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r
    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images 
        
        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        # Convert feature point arrays to matrices
        q1_l_mat = np.matrix(q1_l)
        q1_r_mat = np.matrix(q1_r)
        q2_l_mat = np.matrix(q2_l)
        q2_r_mat = np.matrix(q2_r)
        P_l_mat = np.matrix(self.P_l)
        P_r_mat = np.matrix(self.P_r)
        P_l_mat = P_l_mat.astype(np.float32)
        P_r_mat = P_r_mat.astype(np.float32)
        q1_l_tran = q1_l_mat.T
        q2_l_tran = q2_l_mat.T
        q1_r_tran = q1_r_mat.T
        q2_r_tran = q2_r_mat.T
  
        # Triangulate points from i-1'th image
        Q1_homogeneous = cv2.triangulatePoints(P_l_mat, P_r_mat, q1_l_tran, q1_r_tran)
        # Un-homogenize
        Q1 = np.transpose(Q1_homogeneous[:3] / Q1_homogeneous[3])

        # Triangulate points from i'th image
        Q2_homogeneous = cv2.triangulatePoints(P_l_mat, P_r_mat, q2_l_tran, q2_r_tran)
        # Un-homogenize
        Q2 = np.transpose(Q2_homogeneous[:3] / Q2_homogeneous[3])

        return Q1, Q2

  

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            #print(sample_idx)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not found any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix

    def get_pose(self,old_imgL, old_imgR, new_imgL, new_imgR):
        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # Get the i-1'th image and i'th image
        img1_l = old_imgL
        img2_l = new_imgL


        # Get the tiled keypoints
        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)
        
        # Track the keypoints
        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)


        # Calculate the disparities
        old_disp = np.divide(self.disparity.compute(old_imgL, old_imgR).astype(np.float32), 16)
        new_disp = np.divide(self.disparity.compute(new_imgL, new_imgR).astype(np.float32), 16)

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, old_disp, new_disp )


        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        return transformation_matrix

  


skip_frames=2
data_dir = '' 
vo = VisualOdometry(data_dir)

gt_path = []
estimated_path = []
camera_pose_list = []
start_pose = np.identity (4)

process_frames = False
old_frame_left = None
old_frame_right = None
new_frame_left = None
new_frame_right = None
frame_counter = 0

cur_pose = start_pose

rs = RealsenseCamera()
while True:
   
   # Capture frame-by-frame
    ret, new_frame_left,new_frame_right,*_ = rs.get_frame_stream()
    
   
    frame_counter += 1

    start = time.perf_counter()

    if process_frames and ret:
        try: 

            transf = vo.get_pose(old_frame_left, old_frame_right, new_frame_left, new_frame_right)
            
            cur_pose = cur_pose @ transf

            camera_pose_list.append(cur_pose)
            estimated_path.append((cur_pose [0, 3], cur_pose[1, 3]))

            estimated_camera_pose_x, estimated_camera_pose_y = cur_pose [0, 3], cur_pose [1, 3]

        except Exception as e:
            print("An error occurred:", str(e))
            break
    
    elif process_frames and ret is False:
        break

    old_frame_left = new_frame_left
    old_frame_right = new_frame_right

    process_frames = True

    end = time.perf_counter()
    total_time = end - start
    fps = 1 / total_time
    
    cv2.imshow("imgL", new_frame_left)
    cv2.imshow("imgR", new_frame_right)

    # Press Q on keyboard to exit
    if cv2.waitKey (25) & 0xFF == ord ('q'):
        break

rs.release()
cv2.destroyAllWindows () 

# Plotting #
number_of_frames = len(camera_pose_list) 
image_size = np.array([848, 480])

plt.figure()
ax = plt.axes (projection='3d')


camera_pose_poses = np.array(camera_pose_list)

key_frames_indices = np.linspace(0, len(camera_pose_poses) - 1, number_of_frames, dtype=int)
colors = cycle("rgb")

camera_centers = np.zeros((number_of_frames, 3))

filename = "transformation_matrices.txt"
filenamecent = "cameraCenters.txt"

for i, pose_index in enumerate(key_frames_indices):
    transformation_matrix = camera_pose_poses[pose_index]
    camera_center = transformation_matrix[:3, 3]
    camera_centers[i] = camera_center
    camera_centers_scaled = camera_centers / scale_factor
    with open(filename, "a") as f:
        f.write(f"Transformation Matrix {i+1}:\n")
        np.savetxt(f, transformation_matrix, fmt="%.6f")
        f.write("\n")
        

    with open(filenamecent, "a") as fc:       
        np.savetxt(fc, camera_center ,fmt="%6f")
        fc.write("\n")
        
                      
    
ax.scatter(camera_centers_scaled[:, 0], camera_centers_scaled[:, 1], camera_centers_scaled[:, 2], c='b')
# Calculate the range of all three axes
max_range = np.array([camera_centers_scaled[:, 0].max() - camera_centers_scaled[:, 0].min(),
                    camera_centers_scaled[:, 1].max() - camera_centers_scaled[:, 1].min(),
                    camera_centers_scaled[:, 2].max() - camera_centers_scaled[:, 2].min()]).max() / 2.0

# Calculate the center of all three axes
mid_x = (camera_centers_scaled[:, 0].max() + camera_centers_scaled[:, 0].min()) * 0.5
mid_y = (camera_centers_scaled[:, 1].max() + camera_centers_scaled[:, 1].min()) * 0.5
mid_z = (camera_centers_scaled[:, 2].max() + camera_centers_scaled[:, 2].min()) * 0.5

# Set the limits of all three axes to achieve equal scaling
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)


# Set x, y, z labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.savefig(os.path.join(save_folder, "3D_scatter_plot.png"), format='png')

plt.show()


take_every_th_camera_pose = 2

estimated_path = np.array(estimated_path[::take_every_th_camera_pose])


# Apply the scale factor to the estimated path
estimated_path *= 20
# Calculate the minimum and maximum values for each axis
min_x = np.min(estimated_path[:, 0])
min_y = np.min(estimated_path[:, 1])
max_x = np.max(estimated_path[:, 0])
max_y = np.max(estimated_path[:, 1])

# Calculate the range for each axis
range_x = max_x - min_x
range_y = max_y - min_y
range_max = max(range_x, range_y)

# Calculate the origin point
origin_x = min_x
origin_y = min_y

# Set the x and y axis limits
plt.xlim(origin_x - range_max, origin_x + range_max)  # Set the x-axis limit
plt.ylim(origin_y - range_max, origin_y + range_max)  # Set the y-axis limit



plt.plot(estimated_path[:, 0], estimated_path[:, 1])


# Check the data type and shape of the estimated_path array
print("Data type of estimated_path:", estimated_path.dtype)
print("Shape of estimated_path:", estimated_path.shape)


plt.xlabel('x values from estimated_path')  # Label for X-axis
plt.ylabel('y values from estimated_path')  # Label for Y-axis
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#x-z plot

take_every_th_camera_pose = 2


estimated_path = np.array(estimated_path[::take_every_th_camera_pose])

# Scale the estimated path
estimated_path *= 10

# Extract x and z coordinates from estimated_path
x_values = estimated_path[:, 0]
z_values = estimated_path[:, 1]

# Plot x and z coordinates on the same plot
plt.plot(x_values, z_values)
plt.xlabel('X Coordinate')
plt.ylabel('Z Coordinate')
plt.title('Estimated Path')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed Time:", elapsed_time)
