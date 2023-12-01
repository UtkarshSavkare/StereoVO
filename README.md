# StereoVO

Mater Thesis Project EITM 2021-23 Technische Universität Wien (TUW) & Scuola Universitaria Professionale Della Svizzera Italiana (SUPSI)
The vision-based motion estimation system is designed to estimate the motion and pose of a mobile robot in harsh environments.
The system uses stereo visual odometry techniques to analyse image sequences from a stereo camera setup. 

# Modules 

Stereo Camera Calibration 
Stereo_calibration/calibration_acquireimg.py used to acquire and save images for stereo calibration from Intel D455
Stereo_calibration/stereovision_calibration.py  performs stereo camera calibration using a set of detected chessboard corners from the left and right stereo images.

Offline_KITTI
The KITTI Visual Odometry dataset is a well-established benchmark designed to assess the accuracy and robustness of visual odometry algorithms under various real-world driving scenarios.
we have utilized the grayscale sequence 01 from the KITTI Visual Odometry dataset, encompassing the first 101 frames to understand the basic working of VO algorithm.
we explored monocular VO - Offline_KITTI/mono_VO_Offline.py
we explored Stereo VO - Offline_KITTI/stereo_VO_Offline.py

Online_SVO
Finally we develop a custumised algorithm for Stereo VO in outdoor environment and perform real-time experiments agianst ground truth from Leica Absolute Tracker AT960-MR.
Online_SVO/videostream_D455.py used to capture sequence of stereo pairs.
Online_SVO/scale_factor.py to get appropriate scale factor.
Online_SVO/SVO_Main.py perform camera pose and motion estimation (SVO).

# Results
Detailed analysis in published master thesis at TU Wien Online repository https://repositum.tuwien.at/
