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
from cv_bridge import CvBridge, CvBridgeError

from . HALIntelT265 import HALIntelT265 
from intel_t265_interfaces.msg import BinocularView  

import numpy as np
import scipy.misc


class Ros2HALIntelT265(Node):

    def __init__(self):
        super().__init__('hal_intel_t265')
        self.publisher_raw_images = self.create_publisher(BinocularView, '/intel_t265/binocular_images', 10)

        #self.publisher_rect_images = self.create_publisher(BinocularView, '/intel_t265/rectified_binocular_images', 10)

        #self.publisher_disp_map = self.create_publisher(Image, '/intel_t265/disparity_map', 10)

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.cam = HALIntelT265()

        self.bridge = CvBridge()


    def timer_callback(self):

        image_left, image_right = self.cam.get_stereo_view()

        if image_left is not None and image_right is not None:
            msg = BinocularView()
            msg.left_image =  self.bridge.cv2_to_imgmsg(image_left)
            msg.right_image = self.bridge.cv2_to_imgmsg(image_right)
            self.publisher_raw_images.publish(msg)

            '''
            self.cam.undistort()
            msg2 = BinocularView()
            msg2.left_image =  self.bridge.cv2_to_imgmsg(color_image_left)
            msg2.right_image = self.bridge.cv2_to_imgmsg(color_image_right)
            self.publisher_rect_images.publish(msg2)

            self.cam.compute_disparity()
            msg3 = Image()
            msg3 =  self.bridge.cv2_to_imgmsg(self.cam.disparity_map)

            self.publisher_disp_map.publish(msg3)
            '''



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
