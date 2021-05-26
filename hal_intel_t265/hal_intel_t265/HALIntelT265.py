# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pyrealsense2 as rs

import cv2
import numpy as np
from math import tan, pi


# Set up a mutex to share data between threads 
from threading import Lock


class HALIntelT265:

    def __init__(self):
        
        self.frame_mutex = Lock()
        self.frame_data = {"left"  : None,
                      "right" : None,
                      "timestamp_ms" : None
                      }

        # Declare RealSense pipeline, encapsulating the actual device and sensors
        self.pipe = rs.pipeline()

        # Build config object and stream everything
        self.cfg = rs.config()

        # Start streaming with our callback
        self.pipe.start(self.cfg, self.callback)

        # Retreive the stream and intrinsic properties for both cameras
        self.profiles = self.pipe.get_active_profile()
        self.streams = {"left"  : self.profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
                        "right" : self.profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}

        self.intrinsics = { "left"  : self.streams["left"].get_intrinsics(),
                            "right" : self.streams["right"].get_intrinsics()}

        # Print information about both cameras
        print("Left camera:",  self.intrinsics["left"])
        print("Right camera:", self.intrinsics["right"])

        # Translate the intrinsics from librealsense into OpenCV
        self.K_left  = self.camera_matrix(self.intrinsics["left"])
        self.D_left  = self.fisheye_distortion(self.intrinsics["left"])
        self.K_right = self.camera_matrix(self.intrinsics["right"])
        self.D_right = self.fisheye_distortion(self.intrinsics["right"])
        (self.width, self.height) = (self.intrinsics["left"].width, self.intrinsics["left"].height)

        # Get the relative extrinsics between the left and right camera
        (self.R, self.T) = self.get_extrinsics(self.streams["left"], self.streams["right"])


        # Configure the OpenCV stereo algorithm. See
        # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html for a
        # description of the parameters
        self.window_size = 5
        self.min_disp = 0
        # must be divisible by 16
        self.num_disp = 112 - self.min_disp
        self.max_disp = self.min_disp + self.num_disp
        self.stereo = cv2.StereoSGBM_create(minDisparity = self.min_disp,
                                       numDisparities = self.num_disp,
                                       blockSize = 16,
                                       P1 = 8*3*self.window_size**2,
                                       P2 = 32*3*self.window_size**2,
                                       disp12MaxDiff = 1,
                                       uniquenessRatio = 10,
                                       speckleWindowSize = 100,
                                       speckleRange = 32)


        # We need to determine what focal length our undistorted images should have
        # in order to set up the camera matrices for initUndistortRectifyMap.  We
        # could use stereoRectify, but here we show how to derive these projection
        # matrices from the calibration and a desired height and field of view

        # We calculate the undistorted focal length:
        #
        #         h
        # -----------------
        #  \      |      /
        #    \    | f  /
        #     \   |   /
        #      \ fov /
        #        \|/
        self.stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov
        self.stereo_height_px = 300          # 300x300 pixel stereo output
        self.stereo_focal_px = self.stereo_height_px/2 / tan(self.stereo_fov_rad/2)

        # We set the left rotation to identity and the right rotation
        # the rotation between the cameras
        self.R_left = np.eye(3)
        self.R_right = self.R

        # The stereo algorithm needs max_disp extra pixels in order to produce valid
        # disparity on the desired output region. This changes the width, but the
        # center of projection should be on the center of the cropped image
        self.stereo_width_px = self.stereo_height_px + self.max_disp
        self.stereo_size = (self.stereo_width_px, self.stereo_height_px)
        self.stereo_cx = (self.stereo_height_px - 1)/2 + self.max_disp
        self.stereo_cy = (self.stereo_height_px - 1)/2

        # Construct the left and right projection matrices, the only difference is
        # that the right projection matrix should have a shift along the x axis of
        # baseline*focal_length
        self.P_left = np.array([[self.stereo_focal_px,       0,              self.stereo_cx,  0],
                                [0,               self.stereo_focal_px, self.stereo_cy,  0],
                                [0,                     0,                  1,           0]])
        self.P_right = self.P_left.copy()
        self.P_right[0][3] = self.T[0]*self.stereo_focal_px


        # Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
        # since we will crop the disparity later
        self.Q = np.array([[1, 0,       0, -(self.stereo_cx - self.max_disp)],
                           [0, 1,       0, -self.stereo_cy],
                           [0, 0,       0, self.stereo_focal_px],
                           [0, 0, -1/self.T[0], 0]])

        # Create an undistortion map for the left and right camera which applies the
        # rectification and undoes the camera distortion. This only has to be done
        # once
        self.m1type = cv2.CV_32FC1
        (self.lm1, self.lm2) = cv2.fisheye.initUndistortRectifyMap(self.K_left, self.D_left, self.R_left, 
                                                                    self.P_left, self.stereo_size, self.m1type)

        (self.rm1, self.rm2) = cv2.fisheye.initUndistortRectifyMap(self.K_right, self.D_right, self.R_right, 
                                                                    self.P_right, self.stereo_size, self.m1type)
        self.undistort_rectify = {"left"  : (self.lm1, self.lm2),
                                    "right" : (self.rm1, self.rm2)}

        self.mode = "stack"
    

    def __del__(self):
        self.pipe.stop()

    def acquire_data(self):
        # Check if the camera has acquired any frames
        self.frame_mutex.acquire()
        self.valid = self.frame_data["timestamp_ms"] is not None
        self.frame_mutex.release()

        # If frames are ready to process
        if self.valid:
            # Hold the mutex only long enough to copy the stereo frames
            self.frame_mutex.acquire()
            self.frame_copy = {"left"  : self.frame_data["left"].copy(),
                                "right" : self.frame_data["right"].copy()}
            self.frame_mutex.release()

    def undistort(self):
            # Undistort and crop the center of the frames
            self.center_undistorted = {"left" : cv2.remap(src = self.frame_copy["left"],
                                          map1 = self.undistort_rectify["left"][0],
                                          map2 = self.undistort_rectify["left"][1],
                                          interpolation = cv2.INTER_LINEAR),
                                  "right" : cv2.remap(src = self.frame_copy["right"],
                                          map1 = self.undistort_rectify["right"][0],
                                          map2 = self.undistort_rectify["right"][1],
                                          interpolation = cv2.INTER_LINEAR)}

            self.color_image_left = cv2.cvtColor(self.center_undistorted["left"][:,self.max_disp:], cv2.COLOR_GRAY2RGB)

            self.color_image_right = cv2.cvtColor(self.center_undistorted["right"][:,self.max_disp:], cv2.COLOR_GRAY2RGB)

    def compute_disparity(self):
            # compute the disparity on the center of the frames and convert it to a pixel disparity (divide by DISP_SCALE=16)
            self.disparity = self.stereo.compute(self.center_undistorted["left"], 
                                        self.center_undistorted["right"]).astype(np.float32) / 16.0

            # re-crop just the valid part of the disparity
            self.disparity = self.disparity[:,self.max_disp:]

            # convert disparity to 0-255 and color it
            self.disp_vis = 255*(self.disparity - self.min_disp)/ self.num_disp
            self.disp_color = cv2.applyColorMap(cv2.convertScaleAbs(self.disp_vis,1), cv2.COLORMAP_JET)
            self.color_image_left = cv2.cvtColor(self.center_undistorted["left"][:,self.max_disp:], cv2.COLOR_GRAY2RGB)

            self.color_image_right = cv2.cvtColor(self.center_undistorted["right"][:,self.max_disp:], cv2.COLOR_GRAY2RGB)

            self.disparity_map = self.color_image_left.copy()
            self.ind = self.disparity >= self.min_disp
            self.disparity_map[self.ind, 0] = self.disp_color[self.ind, 0]
            self.disparity_map[self.ind, 1] = self.disp_color[self.ind, 1]
            self.disparity_map[self.ind, 2] = self.disp_color[self.ind, 2]
            
    def get_full_stack(self):

        self.acquire_data()
        self.undistort()
        self.compute_disparity()

        if not self.valid:
            return None 

        if self.mode == "stack":
            return np.hstack((self.color_image_left, self.color_image_right))

        else:
            return self.disparity_map

    def get_stereo_view(self):

        self.acquire_data()

        if not self.valid:
            return None, None

        return self.frame_copy["left"], self.frame_copy["right"]


    """
    Returns R, T transform from src to dst
    """
    def get_extrinsics(self, src, dst):
        extrinsics = src.get_extrinsics_to(dst)
        R = np.reshape(extrinsics.rotation, [3,3]).T
        T = np.array(extrinsics.translation)
        return (R, T)

    """
    Returns a camera matrix K from librealsense intrinsics
    """
    def camera_matrix(self, intrinsics):
        return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                         [            0, intrinsics.fy, intrinsics.ppy],
                         [            0,             0,              1]])

    """
    Returns the fisheye distortion from librealsense intrinsics
    """
    def fisheye_distortion(self, intrinsics):
        return np.array(intrinsics.coeffs[:4])

    """
    This callback is called on a separate thread, so we must use a mutex
    to ensure that data is synchronized properly. We should also be
    careful not to do much work on this thread to avoid data backing up in the
    callback queue.
    """
    def callback(self, frame):
        if frame.is_frameset():
            frameset = frame.as_frameset()
            f1 = frameset.get_fisheye_frame(1).as_video_frame()
            f2 = frameset.get_fisheye_frame(2).as_video_frame()
            left_data = np.asanyarray(f1.get_data())
            right_data = np.asanyarray(f2.get_data())
            ts = frameset.get_timestamp()
            self.frame_mutex.acquire()
            self.frame_data["left"] = left_data
            self.frame_data["right"] = right_data
            self.frame_data["timestamp_ms"] = ts
            self.frame_mutex.release()

