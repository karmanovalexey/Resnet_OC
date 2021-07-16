#!/usr/bin/env python3

import rospy
import torch
import cv_bridge
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.segmentation import fcn_resnet50
import torch.nn.functional as F
from sensor_msgs.msg import Image
from segformer.segformer import get_model

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class SegmentatorNode:

    def __init__(self):
        rospy.init_node('segmentator')

        rospy.loginfo('Loading model...')
        # self.model = fcn_resnet50(pretrained=True)
        self.model = get_model('/home/docker_segmentator/catkin_ws/src/segmentator/saves')
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        rospy.loginfo('Model loaded')

        self.sub = rospy.Subscriber('image', Image, self.on_image, queue_size=10)
        self.pub = rospy.Publisher('segmentation', Image, queue_size=10)

        self.br = cv_bridge.CvBridge()


    def on_image(self, image_msg : Image):
        image = self.br.imgmsg_to_cv2(image_msg, 'rgb8')
        print(image.shape)
        batch = SegmentatorNode.preproc_cv2(image)
        if torch.cuda.is_available():
            batch = batch.cuda()
        logits = self.model(batch)

        probs = torch.softmax(logits['aux'][0], 0)
        segm = probs.argmax(dim=0) * (probs.max(dim=0).values > 0.5)

        segm_msg = self.br.cv2_to_imgmsg(segm.cpu().numpy().astype(np.uint8), 'mono8')
        segm_msg.header = image_msg.header
        self.pub.publish(segm_msg)


    @staticmethod
    def preproc_cv2(image):
        image_tensor = torch.Tensor(image.copy()).float() / 255
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        image_tensor = resize(
            input=image_tensor,
            size=(600,1200),
            mode='bilinear',
            align_corners=False)
        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.permute(2, 0, 1)

        batch = image_tensor.unsqueeze(0)

        return batch


    def spin(self):
        rospy.spin()


def main():
    node = SegmentatorNode()
    node.spin()


if __name__ == '__main__':
    main()
