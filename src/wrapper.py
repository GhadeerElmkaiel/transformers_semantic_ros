#! /usr/bin/env python

from __future__ import print_function

import os
import sys
import rospy

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch

from torchvision import transforms
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.visualize import get_color_pallete
from segmentron.config import cfg

import cv2
import numpy as np
from PIL import Image as Img

from sensor_msgs.msg import Image
#from cv_bridge import CvBridge

BPP = {
  'bgr8': 3,
  'rgb8': 3,
  '16UC1': 2,
  '8UC1': 1,
  'mono16': 2,
  'mono8': 1
}

class SegmentationWrapper:
    """
    Wrapper for glass segmentation under ROS
    """
    def __init__(self):
        """
        Initializer
        """
        # TOPIC_IMAGE = rospy.get_param("image_topic","/camera_b/color/image_raw")   
        # TOPIC_IMAGE = rospy.get_param("image_topics","/zed2/zed_node/rgb/image_rect_color")
        self.TOPICS_IMAGE = rospy.get_param("image_topics",["/zed2/zed_node/rgb/image_rect_color"])
        # self.TOPICS_IMAGE = rospy.get_param("image_topics",["/camera_h/color/image_raw"])
        self.TOPIC_SEMANTIC = rospy.get_param("mask_topic","/segmentation")
        self.TOPIC_ORIGINAL = rospy.get_param("orig_topic","/orig_image")
        self.CKPT_PATH = rospy.get_param("ckpt_path","/demo")
        self.ROOT_DIR = os.path.abspath(os.curdir) + "/src/transformers_semantic_ros/src"

        ##################################### Trans2Seg Configs ####################################
        self.MODEL_NAME = rospy.get_param("model_name","Trans2Seg")
        self.CKPT_NAME = rospy.get_param("ckpt_name","trans2seg_model.pth")
        self.CONF_PATH = rospy.get_param("config_path","/configs/trans10kv2/trans2seg/trans2seg_medium_all_sber.yaml")
        self.LIST_OF_CLASSES = rospy.get_param("list_of_classes",["Mirror", "Glass", "Floor Under Obstacles", "Floor", "Other Reflective Surfaces"])
        ############################################################################################
        
        ##################################### TransLab Configs #####################################
        # self.MODEL_NAME = rospy.get_param("model_name","TransLab")
        # self.CKPT_NAME = rospy.get_param("ckpt_name","translab_model.pth")
        # self.CONF_PATH = rospy.get_param("config_path","/configs/translab/translab_bs4.yaml")
        # self.LIST_OF_CLASSES = rospy.get_param("list_of_classes",["All_Optical", "Floor"])
        ############################################################################################

        self.model_path = self.ROOT_DIR+self.CKPT_PATH+"/"+self.CKPT_NAME
        self.config_file = self.ROOT_DIR+self.CONF_PATH

        self.resize_image_to = (520,  520)
        if self.MODEL_NAME == "Trans2Seg":
            self.resize_image_to = (512,  512)


        cfg.update_from_file(self.config_file)
        cfg.TEST.TEST_MODEL_PATH = self.model_path
        cfg.PHASE = 'test'
        cfg.MODEL.MODEL_NAME = self.MODEL_NAME
        cfg.check_and_freeze()

        self.topics_msgs = {}
        self.image_subscribers = {}
        self.mask_publishers = {}
        self.image_publishers = {}
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu" 
        self.device = torch.device(dev)

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        if self.MODEL_NAME== "TransLab" or self.MODEL_NAME=="Trans2Seg":				# If model is TransLab or Trans2Seg
            self.model = get_segmentation_model().to(self.device)	
        else:
            #model = get_segmentation_model().to(device)
            pass
        self.model.to(self.device)
        self.model.eval()

        for topic in self.TOPICS_IMAGE:
            self.topics_msgs[topic] = None
            self.image_subscribers[topic] = rospy.Subscriber(topic, Image, self.onImageCB, callback_args=(topic), queue_size = 1)
            self.mask_publishers[topic] = rospy.Publisher(self.TOPIC_SEMANTIC+topic, Image, queue_size = 1)
            self.image_publishers[topic] = rospy.Publisher(self.TOPIC_ORIGINAL+topic, Image, queue_size = 1)
            rospyLogInfoWrapper("Adding segmentation process for the topic: "+topic)
        # self.test_pub = rospy.Publisher("Test", Image, queue_size = 1)
    
    def onImageCB(self, msg, args):
        topic = str(args)
        rospyLogDebugWrapper("Recived new Image, topic: "+topic)
        arr = self.imgmsgToCV2(msg)
        img = Img.fromarray(arr)
        size_ = img.size
        
        img = img.resize(self.resize_image_to, Img.BILINEAR)
        img_arr = np.array(img)
        self.topics_msgs[topic] = {"msg":msg, "size":size_, "img_arr": img_arr}

        tensor = self.input_transform(img_arr)
        tensor = torch.unsqueeze(tensor,0).to(self.device)
        
        # rospyLogInfoWrapper("Tensor size"+ str(tensor.shape))
        with torch.no_grad():
            # pass
            if self.MODEL_NAME == "TransLab":
                output, output_boundary = self.model.evaluate(tensor)
                result = output.argmax(1)[0].data.cpu().numpy().astype('uint8')*127
                result = cv2.resize(result, size_, interpolation=cv2.INTER_NEAREST)
                encode = "mono8"
            elif self.MODEL_NAME == "Trans2Seg":
                output = self.model(tensor)
                mask = torch.argmax(output[0], 1)[0].cpu().data.numpy()
                result = get_color_pallete(mask, cfg.DATASET.NAME)
                result = result.convert("RGB")
                result = result.resize(size_)
                result = np.array(result)
                encode = "rgb8"
                # rospyLogInfoWrapper(str(result))

            # result = output.argmax(1)[0].data.cpu().numpy().astype('uint8')*127
            # result = cv2.resize(result, size_, interpolation=cv2.INTER_NEAREST)
            #TODO
            # Create a image msg for the masks and original image for each single topic images [N, 3 , W, H]
            semantic_msg = self.CV2ToImgmsg(result, encoding=encode)
            # test_msg = Image()
            # self.test_pub.publish(semantic_msg)
            self.mask_publishers[topic].publish(semantic_msg)
            self.image_publishers[topic].publish(msg)

    def imgmsgToCV2(self, data, desired_encoding="passthrough", flip_channels=False):
        """
        Converts a ROS image to an OpenCV image without using the cv_bridge package,
        for compatibility purposes.
        """

        if desired_encoding == "passthrough":
            encoding = data.encoding
        else:
            encoding = desired_encoding

        if encoding == 'bgr8' or (encoding=='rgb8' and flip_channels):
            return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 3))
        elif encoding == 'rgb8' or (encoding=='bgr8' and flip_channels):
            return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 3))[:, :, ::-1]
        elif encoding == 'mono8' or encoding == '8UC1':
            return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width))
        elif encoding == 'mono16' or encoding == '16UC1':
            return np.frombuffer(data.data, np.uint16).reshape((data.height, data.width))
        elif encoding == 'bgra8' or (encoding=='rgba8' and flip_channels):
            return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 4))[:, :, :-1][:, :, ::-1]
        elif encoding == 'rgba8' or (encoding=='bgra8' and flip_channels):
            return np.frombuffer(data.data, np.uint8).reshape((data.height, data.width, 4))[:, :, :-1]
        else:
            rospy.logwarn("Unsupported encoding %s" % encoding)
            return None

    def CV2ToImgmsg(self, cv2img, encoding='mono8'):
        """
        Converts an OpenCV image to a ROS image without using the cv_bridge package,
        for compatibility purposes.
        """

        msg = Image()
        msg.width = cv2img.shape[1]
        msg.height = cv2img.shape[0]
        
        msg.encoding = encoding
        msg.step = BPP[encoding]*cv2img.shape[1]
        msg.data = np.ascontiguousarray(cv2img).tobytes()

        return msg


def rospyLogInfoWrapper(text):
    """Wrapper for rospy.loginfo

    Args:
        text (str): Text to send via rospy.loginfo
    """
    wrapped_text = rospy.get_name().lstrip('/') + ': ' + str(text)
    rospy.loginfo(wrapped_text)

def rospyLogWarnWrapper(text):
    """Wrapper for rospy.logwarn

    Args:
        text (str): Text to send via rospy.logwarn
    """
    wrapped_text = rospy.get_name().lstrip('/') + ': ' + str(text)
    rospy.logwarn(wrapped_text)

def rospyLogErrWrapper(text):
    """Wrapper for rospy.logerr

    Args:
        text (str): Text to send via rospy.logerr
    """
    wrapped_text = rospy.get_name().lstrip('/') + ': ' + str(text)
    rospy.logerr(wrapped_text)

def rospyLogDebugWrapper(text):
    """Wrapper for rospy.logdebug

    Args:
        text (str): Text to send via rospy.logdebug
    """
    wrapped_text = rospy.get_name().lstrip('/') + ': ' + str(text)
    rospy.logdebug(wrapped_text)

