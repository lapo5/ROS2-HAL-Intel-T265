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

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

from . HALIntelT265 import HALIntelT265 

import numpy as np
import scipy.misc


class Ros2HALIntelT265(Node):

    def __init__(self):
        super().__init__('hal_intel_t265')

        self.publisher_raw_image_left = self.create_publisher(Image, '/intel_t265/left_rect', 1)
        self.publisher_raw_image_right = self.create_publisher(Image, '/intel_t265/right_rect', 1)

        self.publisher_disp_map = self.create_publisher(Image, '/intel_t265/disparity_map', 1)

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.cam = HALIntelT265(mode="separated_rect_with_disparity")

        self.bridge = CvBridge()


    def timer_callback(self):

        cam_data = self.cam.get_full_stack()

        if cam_data['left'] is not None and cam_data['right'] is not None:
            
            left_image_msg =  self.bridge.cv2_to_imgmsg(cam_data['left'])
            left_image_msg.header = Header()
            left_image_msg.header.stamp = self.get_clock().now().to_msg()
            left_image_msg.header.frame_id = "left_cam"
            self.publisher_raw_image_left.publish(left_image_msg)

            right_image_msg = self.bridge.cv2_to_imgmsg(cam_data['right'])
            right_image_msg.header = Header()
            right_image_msg.header.stamp = self.get_clock().now().to_msg()
            right_image_msg.header.frame_id = "right_cam"
            self.publisher_raw_image_right.publish(right_image_msg)

            disp_msg =  self.bridge.cv2_to_imgmsg(cam_data['disp'])
            disp_msg.header = Header()
            disp_msg.header.stamp = self.get_clock().now().to_msg()
            disp_msg.header.frame_id = "left_cam"

            self.publisher_disp_map.publish(disp_msg)



def main(args=None):
    rclpy.init(args=args)

    hal_intel = Ros2HALIntelT265()

    rclpy.spin(hal_intel)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hal_intel.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
