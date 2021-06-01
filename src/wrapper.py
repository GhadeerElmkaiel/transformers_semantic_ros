#! /usr/bin/env python

from __future__ import print_function

import os
import sys
import time
import rospy
import message_filters

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
        # self.TOPICS_IMAGE = rospy.get_param("~image_topics",["/zed2/zed_node/rgb/image_rect_color"])
        # self.TOPICS_DEPTH = rospy.get_param("~depth_topics",["/zed2/zed_node/rgb/image_rect_color"])
        self.TOPICS_IMAGE = rospy.get_param("~image_topics",["/camera_h/color/image_raw"])
        self.TOPICS_DEPTH = rospy.get_param("~depth_topics",["/camera_h/aligned_depth_to_color/image_raw"])
        self.TOPIC_SEMANTIC = rospy.get_param("~mask_topic","/segmentation")
        self.TOPIC_ORIGINAL = rospy.get_param("~orig_topic","/orig_image")
        self.ROOT_DIR = rospy.get_param("~root_dir", os.path.abspath(os.curdir) + "/src/transformers_semantic_ros/src")
        self.USE_DEPTH = rospy.get_param("~use_depth",True)                          # To choose whether to synchronize depth with the mask
        self.USE_CROP = rospy.get_param("~use_crop",True)                            # To choose whether to use cropped images too to generate the mask
        self.RANDOM_CROP = rospy.get_param("~random_crop",False)                     # To choose whether to use random crop positions
        self.CROP_EDGES = rospy.get_param("~crop_edges",[[0.25, 0.35, 0.75, 0.95]])    # The edges of the cropped image

        self.CKPT_PATH = rospy.get_param("~ckpt_path","/demo")
        # self.ROOT_DIR = os.path.abspath(os.curdir)

        ##################################### Trans2Seg Configs ####################################
        self.MODEL_NAME = rospy.get_param("~model_name","Trans2Seg")
        self.CKPT_NAME = rospy.get_param("~ckpt_name","trans2seg_model.pth")
        self.CONF_PATH = rospy.get_param("~config_path","/configs/Trans2Seg/trans2seg_medium_all_sber.yaml")
        self.LIST_OF_CLASSES = rospy.get_param("~list_of_classes",["Mirror", "Glass", "Floor Under Obstacles", "Floor", "Other Reflective Surfaces"])
        ############################################################################################
        
        ##################################### TransLab Configs #####################################
        # self.MODEL_NAME = rospy.get_param("model_name","TransLab")
        # self.CKPT_NAME = rospy.get_param("ckpt_name","translab_model.pth")
        # self.CONF_PATH = rospy.get_param("config_path","/configs/TransLab/translab_bs4.yaml")
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
        self.depth_subscribers = {}
        

        self.mask_publishers = {}
        self.image_publishers = {}
        self.depth_publishers = {}

        self.sync_img_dpth_subscribers = {}


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
            rospyLogInfoWrapper("Loaded the model!")
        else:
            #model = get_segmentation_model().to(device)
            pass
        self.model.to(self.device)
        self.model.eval()
        self.softmax_layer = torch.nn.Softmax(dim=0)

        if self.USE_DEPTH:                      # To synchronize the depth with the RGB image for further use
            for i, topic in enumerate(self.TOPICS_IMAGE):
                self.topics_msgs[topic] = None
                self.image_subscribers[topic] = message_filters.Subscriber(topic, Image)
                self.depth_subscribers[topic] = message_filters.Subscriber(self.TOPICS_DEPTH[i], Image)
                self.sync_img_dpth_subscribers[topic] =  message_filters.TimeSynchronizer([self.image_subscribers[topic], self.depth_subscribers[topic]], 1)
                self.sync_img_dpth_subscribers[topic].registerCallback(self.onImageSyncCB, topic, )
                # self.sync_img_dpth_subscribers[topic].registerCallback(self.testCB, topic, )
                self.mask_publishers[topic] = rospy.Publisher(self.TOPIC_SEMANTIC+topic, Image, queue_size = 1)
                self.image_publishers[topic] = rospy.Publisher(self.TOPIC_ORIGINAL+topic, Image, queue_size = 1)
                self.depth_publishers[topic] = rospy.Publisher(self.TOPIC_ORIGINAL+self.TOPICS_DEPTH[i], Image, queue_size = 1)
                rospyLogInfoWrapper("Adding Synchronized segmentation process for the topic: "+topic)


        else:
            for topic in self.TOPICS_IMAGE:
                self.topics_msgs[topic] = None
                self.image_subscribers[topic] = rospy.Subscriber(topic, Image, self.onImageCB, callback_args=(topic), queue_size = 1)
                self.mask_publishers[topic] = rospy.Publisher(self.TOPIC_SEMANTIC+topic, Image, queue_size = 1)
                self.image_publishers[topic] = rospy.Publisher(self.TOPIC_ORIGINAL+topic, Image, queue_size = 1)
                rospyLogInfoWrapper("Adding segmentation process for the topic: "+topic)

        self.test_img_pub = rospy.Publisher("Test_img", Image, queue_size = 1)
        self.test_mask_pub = rospy.Publisher("Test_mask", Image, queue_size = 1)
        self.test_confidence_pub = rospy.Publisher("confidence", Image, queue_size = 1)
    
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



            #TODO
            # Create a image msg for the masks and original image for each single topic images [N, 3 , W, H]
            semantic_msg = self.CV2ToImgmsg(result, encoding=encode)

            #########################################################
            self.mask_publishers[topic].publish(semantic_msg)
            self.image_publishers[topic].publish(msg)
            #########################################################


    def onImageSyncCB(self, rgb_msg, depth_msg, args):
        topic = str(args)
        rospyLogDebugWrapper("Recived new Image, topic: "+topic)
        arr = self.imgmsgToCV2(rgb_msg)
        img = Img.fromarray(arr)
        size_ = img.size
        
        img = img.resize(self.resize_image_to, Img.BILINEAR)
        img_arr = np.array(img)

        self.topics_msgs[topic] = {"msg":rgb_msg, "size":size_, "img_arr": img_arr}

        tensor = self.input_transform(img_arr)
        tensor = torch.unsqueeze(tensor,0).to(self.device)

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

            #TODO
            # Create a image msg for the masks and original image for each single topic images [N, 3 , W, H]
            semantic_msg = self.CV2ToImgmsg(result, encoding=encode)

            stamp = rospy.Time.from_sec(time.time())
            semantic_msg.header.stamp = stamp
            rgb_msg.header.stamp = stamp
            depth_msg.header.stamp = stamp
            self.mask_publishers[topic].publish(semantic_msg)
            self.image_publishers[topic].publish(rgb_msg)
            self.depth_publishers[topic].publish(depth_msg)


    def ImageDepthInfoSyncCB(self, rgb_msg, depth_msg, args):
        topic = str(args)
        rospyLogDebugWrapper("Recived new Image, topic: "+topic)
        arr = self.imgmsgToCV2(rgb_msg)
        img = Img.fromarray(arr)
        size_ = img.size
        
        img = img.resize(self.resize_image_to, Img.BILINEAR)
        img_arr = np.array(img)

        self.topics_msgs[topic] = {"msg":rgb_msg, "size":size_, "img_arr": img_arr}

        tensor = self.input_transform(img_arr)
        tensor = torch.unsqueeze(tensor,0).to(self.device)

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

            #TODO
            # Create a image msg for the masks and original image for each single topic images [N, 3 , W, H]
            semantic_msg = self.CV2ToImgmsg(result, encoding=encode)

            stamp = rospy.Time.from_sec(time.time())
            semantic_msg.header.stamp = stamp
            rgb_msg.header.stamp = stamp
            depth_msg.header.stamp = stamp
            self.mask_publishers[topic].publish(semantic_msg)
            self.image_publishers[topic].publish(rgb_msg)
            self.depth_publishers[topic].publish(depth_msg)


    def testCB(self, rgb_msg, depth_msg, args):
        topic = str(args)
        rospyLogDebugWrapper("Recived new Image, topic: "+topic)
        arr = self.imgmsgToCV2(rgb_msg)
        img = Img.fromarray(arr)
        size_ = img.size


        cropped_arrs = []
        cropped_msgs = []
        cropped_sizes = []
        cropped_edges = []
        cropped_results = []
        cropped_confidences = []
        for crop in self.CROP_EDGES:
            edges = (int(crop[0]*size_[0]), int(crop[1]*size_[1]), int(crop[2]*size_[0]), int(crop[3]*size_[1])) 
            cropped_edges.append(edges)
            cropped = img.crop(edges)
            # cropped_images.append(cropped)
            # cropped = img.crop((0.25*size_[0], 0.35*size_[1], 0.75*size_[0], 0.95*size_[1]))
            cropped_arr = np.array(cropped)
            # rospyLogInfoWrapper(str(cropped_arr.shape))

            cropped_msg = self.CV2ToImgmsg(cropped_arr, encoding="bgr8")
            cropped_msgs.append(cropped_msg)
            size_c = cropped.size
            cropped_sizes.append(size_c)

            cropped = cropped.resize(self.resize_image_to, Img.BILINEAR)

            cropped_arr = np.array(cropped)
            cropped_arrs.append(cropped_arr)


        img = img.resize(self.resize_image_to, Img.BILINEAR)
        img_arr = np.array(img)
        self.topics_msgs[topic] = {"msg":rgb_msg, "size":size_, "img_arr": img_arr}

        tensor = self.input_transform(img_arr)
        tensor = torch.unsqueeze(tensor,0).to(self.device)

        #########################################################
        for cropped_arr in cropped_arrs:
            tensor_c = self.input_transform(cropped_arr)
            tensor_c = torch.unsqueeze(tensor_c,0).to(self.device)

            tensor = torch.cat((tensor, tensor_c))
        #########################################################

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

                output_norm = self.softmax_layer(output[0])
                confidence_all = torch.max(output_norm, 1)[0].cpu().data.numpy()*255
                confidence = np.array(confidence_all, dtype=np.int8)[0]
                conf_img = Img.fromarray(confidence)
                conf_img = conf_img.resize(size_, Img.BILINEAR)
                confidence = np.array(conf_img, dtype=np.int8)

                #########################################################
                for i in range(len(cropped_arrs)):
                    mask_c = torch.argmax(output[0], 1)[1+i].cpu().data.numpy()
                    result_c = get_color_pallete(mask_c, cfg.DATASET.NAME)
                    result_c = result_c.convert("RGB")
                    result_c = result_c.resize(cropped_sizes[i])
                    result_c = np.array(result_c)
                    res = np.zeros_like(result)
                    # rospyLogInfoWrapper("result_c 0: " + str(result_c.shape))
                    # rospyLogInfoWrapper("cropped_sizes 0: " + str(cropped_sizes[i]))
                    # rospyLogInfoWrapper("cropped_edges 0: " + str(cropped_edges[i]))

                    res[cropped_edges[i][1]:cropped_edges[i][3], cropped_edges[i][0]:cropped_edges[i][2],:]=result_c
                    cropped_results.append(res)

                    cropped_confidence = np.array(confidence_all, dtype=np.int8)[i+1]
                    conf_img = Img.fromarray(cropped_confidence)
                    conf_img = conf_img.resize(size_, Img.BILINEAR)
                    cropped_confidence = np.array(conf_img, dtype=np.int8)
                    cropped_confidences.append(cropped_confidence)


            #TODO
            # Create a image msg for the masks and original image for each single topic images [N, 3 , W, H]
            semantic_msg = self.CV2ToImgmsg(result, encoding=encode)

            stamp = rospy.Time.from_sec(time.time())
            semantic_msg.header.stamp = stamp
            rgb_msg.header.stamp = stamp
            depth_msg.header.stamp = stamp
            self.mask_publishers[topic].publish(semantic_msg)
            self.image_publishers[topic].publish(rgb_msg)
            self.depth_publishers[topic].publish(depth_msg)

            #########################################################
            test_mask_msg = self.CV2ToImgmsg(cropped_results[1], encoding=encode)
            test_conf_msg = self.CV2ToImgmsg(cropped_confidences[1], encoding="mono8")

            self.test_img_pub.publish(cropped_msgs[1])
            self.test_mask_pub.publish(test_mask_msg)
            self.test_confidence_pub.publish(test_conf_msg)



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

