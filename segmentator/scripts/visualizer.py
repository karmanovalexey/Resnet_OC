#!/usr/bin/env python3

import rospy
import cv2
import cv_bridge
import numpy as np

from sensor_msgs.msg import Image

import message_filters


class VisualizerNode:

    def __init__(self):
        rospy.init_node('visualizer')

        image_sub = message_filters.Subscriber('image', Image)
        segm_sub = message_filters.Subscriber('segmentation', Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, segm_sub], 10, 5)
        self.ts.registerCallback(self.on_image_segm)

        self.pub = rospy.Publisher('segmentation_color', Image, queue_size=10)

        self.br = cv_bridge.CvBridge()

        PALETTE = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
                [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255],
                [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
                [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
                [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
                [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
                [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
                [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
                [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
                [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
                [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
                [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
                [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
                [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90],
                [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70],
                [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]]
        self.pallete = np.array(PALETTE, dtype=np.uint8)


    def on_image_segm(self, image_msg : Image, segm_msg : Image):
        image = self.br.imgmsg_to_cv2(image_msg)
        segm = self.br.imgmsg_to_cv2(segm_msg)

        segm_color = self.pallete[segm]

        segm_color_msg = self.br.cv2_to_imgmsg(segm_color, 'rgb8')
        segm_color_msg.header = segm_msg.header

        self.pub.publish(segm_color_msg)


    def spin(self):
        rospy.spin()


def main():
    node = VisualizerNode()
    node.spin()


if __name__ == '__main__':
    main()
