#! /usr/bin/env python

from __future__ import print_function

import os
import sys
import time
import random
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
        self.USE_DEPTH = rospy.get_param("~use_depth",True)                             # To choose whether to synchronize depth with the mask
        self.USE_CROP = rospy.get_param("~use_crop",True)                               # To choose whether to use cropped images too to generate the mask
        self.RANDOM_CROP = rospy.get_param("~random_crop",False)                        # To choose whether to use random crop positions
        self.CROP_EDGES = rospy.get_param("~crop_edges",[[0.25, 0.35, 0.75, 0.95], [0.1, 0.45, 0.9, 0.9]])    # The edges of the cropped image
        self.USE_FLIP = rospy.get_param("~use_flip",True)                               # To choose whether to use the flipped image
        self.USE_DYNAMIC = rospy.get_param("~use_dynamic",True)                         # To choose whether to use the dynamic cropping using the previouse mask
        self.USE_MERGE_FUNCTION = rospy.get_param("~merge_crops_func","And")            # The function responsible for merging different crops 
        self.NUMBER_OF_CROPS = rospy.get_param("~num_of_crops",2)                      # The number of image's crops to use when using random cropping


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

        self.CREATED_RANDOM_CROPS = False

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
                if self.USE_CROP or self.USE_FLIP:
                    self.sync_img_dpth_subscribers[topic].registerCallback(self.onImageSyncWithCropCB, topic, )
                    # self.sync_img_dpth_subscribers[topic].registerCallback(self.onImageSyncCB, topic, )
                    rospyLogInfoWrapper("Adding Synchronized segmentation process with crops for the topic: "+topic)
                else:
                    self.sync_img_dpth_subscribers[topic].registerCallback(self.onImageSyncCB, topic, )
                    rospyLogInfoWrapper("Adding Synchronized segmentation process for the topic: "+topic)
                self.mask_publishers[topic] = rospy.Publisher(self.TOPIC_SEMANTIC+topic, Image, queue_size = 1)
                self.image_publishers[topic] = rospy.Publisher(self.TOPIC_ORIGINAL+topic, Image, queue_size = 1)
                self.depth_publishers[topic] = rospy.Publisher(self.TOPIC_ORIGINAL+self.TOPICS_DEPTH[i], Image, queue_size = 1)


        else:
            for topic in self.TOPICS_IMAGE:
                self.topics_msgs[topic] = None
                self.image_subscribers[topic] = rospy.Subscriber(topic, Image, self.onImageCB, callback_args=(topic), queue_size = 1)
                self.mask_publishers[topic] = rospy.Publisher(self.TOPIC_SEMANTIC+topic, Image, queue_size = 1)
                self.image_publishers[topic] = rospy.Publisher(self.TOPIC_ORIGINAL+topic, Image, queue_size = 1)
                rospyLogInfoWrapper("Adding segmentation process for the topic: "+topic)

        # self.test_img_pub = rospy.Publisher("Test_img", Image, queue_size = 1)
        # self.test_confidence_pub = rospy.Publisher("confidence", Image, queue_size = 1)
        self.test_mask_pub = rospy.Publisher("Test_mask", Image, queue_size = 1)
        self.test_cropped_mask_pub = rospy.Publisher("Test_cropped_mask", Image, queue_size = 1)
    
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
             
                # rospyLogInfoWrapper("Before palette shape "+ str(mask.shape))
                # rospyLogInfoWrapper("Before palette mask "+ str(mask.dtype))
                # rospyLogInfoWrapper("Before palette size "+ str(mask.size))
                # rospyLogInfoWrapper("Before palette nbytes "+ str(mask.nbytes))
                # rospyLogInfoWrapper("Before palette ndim "+ str(mask.ndim))
                # rospyLogInfoWrapper("Before palette flags "+ str(mask.flags))
                result = get_color_pallete(mask, cfg.DATASET.NAME)

                # rospyLogInfoWrapper("After palette shape "+ str(result.size))

                result = result.convert("RGB")
                # result = result.resize(size_)
                result = result.resize(size_, Img.NEAREST)
                result = np.array(result)

                # rospyLogInfoWrapper("After RGB shape "+ str(result.shape))
                # rospyLogInfoWrapper("After RGB mask "+ str(result[0]))

                encode = "rgb8"

            #TODO
            # Create a image msg for the masks and original image for each single topic images [N, 3 , W, H]
            semantic_msg = self.CV2ToImgmsg(result, encoding=encode)

            # stamp = rospy.Time.from_sec(time.time())
            stamp = rgb_msg.header.stamp
            semantic_msg.header.stamp = stamp
            rgb_msg.header.stamp = stamp
            depth_msg.header.stamp = stamp
            self.mask_publishers[topic].publish(semantic_msg)
            self.image_publishers[topic].publish(rgb_msg)
            self.depth_publishers[topic].publish(depth_msg)


    def onImageSyncWithCropCB(self, rgb_msg, depth_msg, args):
        topic = str(args)
        arr = self.imgmsgToCV2(rgb_msg)

        img = Img.fromarray(arr)
        size_ = img.size

        if self.RANDOM_CROP and not self.CREATED_RANDOM_CROPS:
            self.CREATED_RANDOM_CROPS = True
            self.CROP_EDGES = []
            min_x_size = self.resize_image_to[0]/size_[0]
            min_y_size = self.resize_image_to[0]/size_[1]
            for i in range(self.NUMBER_OF_CROPS):
                x_start = random.uniform(0.0, 1.0-min_x_size)
                y_start = random.uniform(0.0, 1.0-min_y_size)
                x_end = random.uniform(x_start+min_x_size, 1.0)
                y_end = random.uniform(y_start+min_y_size, 1.0)

                self.CROP_EDGES.append([x_start, y_start, x_end, y_end])

        img_resized = img.resize(self.resize_image_to, Img.NEAREST)
        orig_img_arr = np.array(img_resized)


        cropped_arrs = [orig_img_arr]
        cropped_msgs = []
        cropped_sizes = [size_]
        cropped_edges = [[0, 0, size_[0], size_[1]]]
        cropped_results = []
        cropped_confidences = []
        if self.USE_FLIP:
            img_flipped = Img.fromarray(arr)
            img_flipped = img_flipped.transpose(method = Img.FLIP_LEFT_RIGHT)
            img_resized = img_flipped.resize(self.resize_image_to, Img.NEAREST)
            img_arr = np.array(img_resized)
            cropped_arrs.append(img_arr)
            cropped_sizes.append(size_)
            cropped_edges.append([0, 0, size_[0], size_[1]])

        for crop in self.CROP_EDGES:
            edges = (int(crop[0]*size_[0]), int(crop[1]*size_[1]), int(crop[2]*size_[0]), int(crop[3]*size_[1])) 
            cropped_edges.append(edges)
            cropped = img.crop(edges)
            cropped_arr = np.array(cropped)

            rospyLogInfoWrapper("CROP_EDGES shape: "+ str(self.CROP_EDGES))
            cropped_msg = self.CV2ToImgmsg(cropped_arr, encoding="bgr8")
            cropped_msgs.append(cropped_msg)
            size_c = cropped.size
            cropped_sizes.append(size_c)

            cropped = cropped.resize(self.resize_image_to, Img.NEAREST)

            cropped_arr = np.array(cropped)
            cropped_arrs.append(cropped_arr)


        self.topics_msgs[topic] = {"msg":rgb_msg, "size":size_, "img_arr": orig_img_arr}

        tensor = self.input_transform(orig_img_arr)
        tensor = torch.unsqueeze(tensor,0).to(self.device)

        #########################################################
        for i in range(1, len(cropped_arrs)):
            cropped_arr = cropped_arrs[i]
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

                # result = result.convert("RGB")
                # result = Img.fromarray(result)
                result = result.resize(size_, Img.NEAREST)
                result = np.array(result)
                encode = "rgb8"

                output_norm = self.softmax_layer(output[0])
                confidence_all = torch.max(output_norm, 1)[0].cpu().data.numpy()*255
                confidence_orig = np.array(confidence_all, dtype=np.uint8)[0]
                conf_img = Img.fromarray(confidence_orig)
                conf_img = conf_img.resize(size_, Img.NEAREST)
                confidence_orig = np.array(conf_img, dtype=np.uint8)

                cropped_results.append(result)
                cropped_confidences.append(confidence_orig)

                start_index = 1
                if self.USE_FLIP:
                    start_index = 2
                    mask_c = torch.argmax(output[0], 1)[1].cpu().data.numpy()
                    result_c = Img.fromarray(mask_c.astype('uint8'))
                    result_c = result_c.transpose(method = Img.FLIP_LEFT_RIGHT)
                    result_c = result_c.resize(size_, Img.NEAREST)
                    result_c = np.array(result_c)

                    cropped_results.append(result_c)

                    cropped_confidence = np.array(confidence_all, dtype=np.uint8)[1]
                    conf_img = Img.fromarray(cropped_confidence)
                    conf_img = conf_img.transpose(method = Img.FLIP_LEFT_RIGHT)
                    # resize to the original crop size
                    conf_img = conf_img.resize(cropped_sizes[1])
                    conf_arr = np.array(conf_img)
                    cropped_confidences.append(conf_arr)



                #########################################################
                for i in range(start_index, len(cropped_arrs)):
                    # mask_c = torch.argmax(output[0], 1)[i+1].cpu().data.numpy()
                    mask_c = torch.argmax(output[0], 1)[i].cpu().data.numpy()
                    # result_c = get_color_pallete(mask_c, cfg.DATASET.NAME)
                    result_c = Img.fromarray(mask_c.astype('uint8'))
                    # result_c = result_c.convert("RGB")
                    # result_c = Img.fromarray(mask_c)

                    # resize to the original crop size
                    result_c = result_c.resize(cropped_sizes[i], Img.NEAREST)
                    result_c = np.array(result_c)


                    # res = np.zeros_like(mask)
                    # res[cropped_edges[i][1]:cropped_edges[i][3], cropped_edges[i][0]:cropped_edges[i][2],:]=result_c

                    cropped_results.append(result_c)

                    # cropped_confidence = np.array(confidence_all, dtype=np.uint8)[i+1]
                    cropped_confidence = np.array(confidence_all, dtype=np.uint8)[i]
                    conf_img = Img.fromarray(cropped_confidence)
                    # resize to the original crop size
                    conf_img = conf_img.resize(cropped_sizes[i])
                    conf_arr = np.array(conf_img)

                    # cropped_confidence = np.zeros_like(result)
                    # cropped_confidence[cropped_edges[i][1]:cropped_edges[i][3], cropped_edges[i][0]:cropped_edges[i][2],:]=conf_arr

                    cropped_confidences.append(conf_arr)

        
            result_merged = self.getMergedSemanticFromCrops(cropped_results, cropped_confidences, cropped_sizes, cropped_edges, self.USE_MERGE_FUNCTION ,[cropped_sizes[0][1], cropped_sizes[0][0]])
            semantic_msg = self.CV2ToImgmsg(result_merged, encoding=encode)

            
            mask_full_image = get_color_pallete(cropped_results[0], cfg.DATASET.NAME)
            mask_full_image = mask_full_image.convert("RGB")
            mask_full_image = np.array(mask_full_image)
            mask_full_msg = self.CV2ToImgmsg(mask_full_image, encoding=encode)

            # Here the array is multiplied by 5, because the id for background in the used palette is 5
            test_mask_cropped_arr = np.ones([cropped_sizes[0][1], cropped_sizes[0][0]])*5
            idx = 1
            test_mask_cropped_arr[cropped_edges[idx][1]:cropped_edges[idx][3], cropped_edges[idx][0]:cropped_edges[idx][2]] = cropped_results[idx]

            mask_cropped_image = get_color_pallete(test_mask_cropped_arr, cfg.DATASET.NAME)
            mask_cropped_image = mask_cropped_image.convert("RGB")
            mask_cropped_image = np.array(mask_cropped_image)
            mask_cropped_msg = self.CV2ToImgmsg(mask_cropped_image, encoding=encode)
            # rospyLogInfoWrapper("msg width: /n"+str(rgb_msg.width))
            # rospyLogInfoWrapper("msg height: /n"+str(rgb_msg.height))
            # rospyLogInfoWrapper("result_merged shape: /n"+str(result_merged.shape))

            # stamp = rospy.Time.from_sec(time.time())
            stamp = rgb_msg.header.stamp
            semantic_msg.header.stamp = stamp
            rgb_msg.header.stamp = stamp
            depth_msg.header.stamp = stamp
            self.mask_publishers[topic].publish(semantic_msg)
            self.image_publishers[topic].publish(rgb_msg)
            self.depth_publishers[topic].publish(depth_msg)

            mask_full_msg.header.stamp = stamp
            mask_cropped_msg.header.stamp = stamp
            self.test_mask_pub.publish(mask_full_msg)
            self.test_cropped_mask_pub.publish(mask_cropped_msg)


    def getMergedSemanticFromCrops(self, crops_result, crops_confidence, crops_sizes, crops_edges, function, full_size):
        palette_mirror = 0
        palette_glass = 1
        palette_OOS = 3
        palette_floor = 4
        palette_FU = 2
        palette_BG = 5

        #################################################################
        ####################### Using And function ######################
        #################################################################

        if function.lower() == "and":
            # rospyLogInfoWrapper("Using And for merging cropped images")
            orig_glass = crops_result[0]==palette_glass
            orig_mirror = crops_result[0]==palette_mirror
            orig_OOS = crops_result[0]==palette_OOS
            orig_floor = crops_result[0]==palette_floor
            orig_FU = crops_result[0]==palette_FU
            # classes = [np.ones(full_size) for _ in range(max_clasess)]
            # merged_classes = []
            for i in range(1, len(crops_result)):
                cropped_glass = crops_result[i]==palette_glass
                cropped_mirror = crops_result[i]==palette_mirror
                cropped_OOS = crops_result[i]==palette_OOS
                cropped_floor = crops_result[i]==palette_floor
                cropped_FU = crops_result[i]==palette_FU
                cropped_all_optical = np.logical_or(cropped_mirror, cropped_glass)
                cropped_all_optical = np.logical_or(cropped_all_optical, cropped_OOS)
                cropped_all_floor = np.logical_or(cropped_FU, cropped_floor)

                cropped_all_optical_extended = np.ones(full_size)
                cropped_all_optical_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_all_optical

                cropped_all_floor_extended = np.ones(full_size)
                cropped_all_floor_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_all_floor

                orig_glass = np.logical_and(cropped_all_optical_extended, orig_glass)
                orig_mirror = np.logical_and(cropped_all_optical_extended, orig_mirror)
                orig_OOS = np.logical_and(cropped_all_optical_extended, orig_OOS)
                orig_floor = np.logical_and(cropped_all_floor_extended, orig_floor)
                orig_FU = np.logical_and(cropped_all_floor_extended, orig_FU)

            background = np.logical_or(orig_glass, orig_mirror)
            background = np.logical_or(background, orig_OOS)
            background = np.logical_or(background, orig_floor)
            background = np.logical_or(background, orig_FU)
            background = np.logical_not(background)
            res = orig_glass * palette_glass
            res += orig_mirror * palette_mirror
            res += orig_OOS * palette_OOS
            res += orig_floor * palette_floor
            res += orig_FU * palette_FU
            res += background * palette_BG

            res_img = get_color_pallete(res, cfg.DATASET.NAME)

            res_img = res_img.convert("RGB")            
            res_arr = np.array(res_img)

            return res_arr

        #################################################################
        ####################### Using Or function #######################
        #################################################################

        elif function.lower() == "or":
            # rospyLogInfoWrapper("Using And for merging cropped images")
            orig_glass = crops_result[0]==palette_glass
            orig_mirror = crops_result[0]==palette_mirror
            orig_OOS = crops_result[0]==palette_OOS
            orig_floor = crops_result[0]==palette_floor
            orig_FU = crops_result[0]==palette_FU
            # classes = [np.ones(full_size) for _ in range(max_clasess)]
            # merged_classes = []
            for i in range(1, len(crops_result)):
                cropped_glass = crops_result[i]==palette_glass
                cropped_mirror = crops_result[i]==palette_mirror
                cropped_OOS = crops_result[i]==palette_OOS
                cropped_floor = crops_result[i]==palette_floor
                cropped_FU = crops_result[i]==palette_FU
                # cropped_all_optical = np.logical_or(cropped_mirror, cropped_glass)
                # cropped_all_optical = np.logical_or(cropped_all_optical, cropped_OOS)
                # cropped_all_floor = np.logical_or(cropped_FU, cropped_floor)

                cropped_glass_extended = np.zeros(full_size)
                cropped_glass_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_glass

                cropped_mirror_extended = np.zeros(full_size)
                cropped_mirror_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_mirror

                cropped_OOS_extended = np.zeros(full_size)
                cropped_OOS_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_OOS

                cropped_floor_extended = np.zeros(full_size)
                cropped_floor_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_floor

                cropped_FU_extended = np.ones(full_size)
                cropped_FU_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_FU

                orig_glass = np.logical_or(cropped_glass_extended, orig_glass)
                orig_free = np.logical_not(orig_glass)
                orig_mirror = np.logical_and(np.logical_or(cropped_mirror_extended, orig_mirror), orig_free)
                orig_free = np.logical_and(orig_free, np.logical_not(orig_mirror))
                orig_OOS = np.logical_and(np.logical_or(cropped_OOS_extended, orig_OOS), orig_free)
                orig_free = np.logical_and(orig_free, np.logical_not(orig_OOS))
                orig_floor = np.logical_and(np.logical_or(cropped_floor_extended, orig_floor), orig_free)
                orig_FU = np.logical_and(cropped_FU_extended, orig_FU)

            background = np.logical_or(orig_glass, orig_mirror)
            background = np.logical_or(background, orig_OOS)
            background = np.logical_or(background, orig_floor)
            background = np.logical_or(background, orig_FU)
            background = np.logical_not(background)
            res = orig_glass * palette_glass
            res += orig_mirror * palette_mirror
            res += orig_OOS * palette_OOS
            res += orig_floor * palette_floor
            res += orig_FU * palette_FU
            res += background * palette_BG

            res_img = get_color_pallete(res, cfg.DATASET.NAME)

            res_img = res_img.convert("RGB")            
            res_arr = np.array(res_img)

            return res_arr

        #################################################################
        #################### Using confidence values# ###################
        #################################################################

        elif function.lower() == "confidence":

            confidence_extended_all = [crops_confidence[0]]
            results_extended_all = [crops_result[0]]
            for i in range(1, len(crops_confidence)):

                cropped_confidence_extended = np.zeros(full_size)
                cropped_confidence_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = crops_confidence[i]
                confidence_extended_all.append(cropped_confidence_extended)

                cropped_segmentation_extended = np.zeros(full_size)
                cropped_segmentation_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = crops_result[i]
                results_extended_all.append(cropped_segmentation_extended)

            max_confidence_args = np.argsort(confidence_extended_all, axis=0)
            result_sorted_confidence = np.take_along_axis(np.array(results_extended_all), max_confidence_args, axis=0)
            res = result_sorted_confidence[-1]
            # rospyLogInfoWrapper("result_max_confidence shape"+str(result_sorted_confidence.shape))
            # rospyLogInfoWrapper("result_max_confidence[0][1]"+str(max_confidence_args[0][0][0:4]))
            # rospyLogInfoWrapper("result_max_confidence[0][2]"+str(max_confidence_args[2][0][0:4]))

            res_img = get_color_pallete(res, cfg.DATASET.NAME)

            res_img = res_img.convert("RGB")            
            res_arr = np.array(res_img)

            return res_arr


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


sberallpallete = [
    [102, 255, 102],  # Mirror
    [51, 221, 255],   # Glass
    [245, 147, 49],   # FU
    [184, 61, 245],   # Other Optical Surface
    [250, 50, 83],    # Floor
    [0, 0, 0],    
]


