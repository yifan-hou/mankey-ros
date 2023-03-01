#! /usr/bin/env python

import argparse
import os

import cv2
# from cv_bridge import CvBridge, CvBridgeError

import sys, os
mankey_path = os.path.dirname(sys.path[0])
print('mankey_path: ', mankey_path)
sys.path.append(mankey_path)

import mankey.network.inference as inference
from mankey.utils.imgproc import PixelCoord
import numpy as np

from time import perf_counter, sleep

# # The ros staff
# from mankey_ros.srv import *
# import rospy
# from sensor_msgs.msg import RegionOfInterest
# from geometry_msgs.msg import Point
# from cv_bridge import CvBridge, CvBridgeError

class MankeyKeypointDetectionServer(object):

    def __init__(self, network_chkpt_path):
        # The network
        print(network_chkpt_path)
        assert os.path.exists(network_chkpt_path)
        self._network, self._net_config = inference.construct_resnet_nostage(network_chkpt_path)

    def handle_keypoint_request(self, request):
        # Decode the image
        try:
            cv_color = request["rgb_image"]
            # cv_depth = request["depth_image"]
        except CvBridgeError as err:
            print('Image conversion error. Please check the image encoding.')
            print(err.message)
            return self.get_invalid_response()

        # The image is correct, perform inference
        try:
            bbox = request["bounding_box"]
            keypoint_pixels = self.process_request_raw(cv_color, bbox)
        except (RuntimeError, TypeError, ValueError):
            print('The inference is not correct.')
            return self.get_invalid_response()

        # The response
        response = {
            "num_keypoints": keypoint_pixels.shape[1],
            "keypoints_pixels": []
        }
        for i in range(keypoint_pixels.shape[1]):
            point = np.array([keypoint_pixels[0, i], keypoint_pixels[1, i]])
            response["keypoints_pixels"].append(point)
        return response

    def process_request_raw(
            self,
            cv_color,  # type: np.ndarray
            bbox,  # type: RegionOfInterest
    ):  # type: (np.ndarray, np.ndarray, RegionOfInterest) -> np.ndarray
        # Parse the bounding box
        top_left, bottom_right = PixelCoord(), PixelCoord()
        top_left.x = bbox["x_offset"]
        top_left.y = bbox["y_offset"]
        bottom_right.x = bbox["x_offset"] + bbox["width"]
        bottom_right.y = bbox["y_offset"] + bbox["height"]

        # Perform the inference
        imgproc_out = inference.proc_input_img_raw(
            cv_color,
            top_left, bottom_right)
        keypointxy_scaled = inference.inference_resnet_nostage(self._network, imgproc_out)
        keypointxy_realunit = inference.get_keypoint_xy_real_unit(keypointxy_scaled)
        keypointxy_pixel = inference.get_3d_prediction(
            keypointxy_realunit,
            imgproc_out.bbox2patch)
        return keypointxy_pixel

    @staticmethod
    def get_invalid_response():
        response = []
        response.num_keypoints = -1
        return response

def main(visualize):

    # ----------------------------------------------------------------
    model_path = '/home/ANT.AMAZON.COM/yifanhou/git/mankey-ros/mankey/experiment/ckpnt/bin_lip/checkpoint-120.pth'
    detect_keypoint = MankeyKeypointDetectionServer(model_path)

    # Get the test data path
    # test_data_path = '/home/ANT.AMAZON.COM/yifanhou/Documents/keypoint_data/backup/eoat_curve_H10A__4hs10/images'
    test_data_path = '/home/ANT.AMAZON.COM/yifanhou/Documents/keypoint_data/data/jaw_H10A_1bs3/images'
    # test_data_path = '/home/ANT.AMAZON.COM/yifanhou/Documents/keypoint_data/data_atlas_long/feb21_copy_1676938259781/images'
    # test_data_path = '/home/ANT.AMAZON.COM/yifanhou/Documents/keypoint_data/data_atlas_singles/failures'
    # if video path is given, do not pause at every frame
    video_save_path = '/home/ANT.AMAZON.COM/yifanhou/Documents/keypoint_data/test_video'
    # video_save_path = []
    # imgs_save_path = '/home/ANT.AMAZON.COM/yifanhou/Documents/keypoint_data/test_imgs'
    imgs_save_path = []
    # ----------------------------------------------------------------

    if os.path.exists(test_data_path):
        folders_inputs = os.listdir(test_data_path)
    else:
        print('Empty test data folder: ', test_data_path)
        exit()

    videoname = os.path.basename(os.path.dirname(test_data_path))

    has_masks = False
    if 'masks' in folders_inputs:
        print('Found masks folder. Will apply masks before running detection')
        has_masks = True

    if len(video_save_path) > 0:
        video_full_path = os.path.join(video_save_path, videoname + '_test.mp4')
        video = cv2.VideoWriter(video_full_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 10, (720,540))
    start_time = perf_counter()
    img_count = 0
    for img_file_name in sorted(os.listdir(test_data_path)):
        print(img_file_name)
        if img_count == 0:
            start_time = perf_counter()
        img_count += 1

        # Read the image
        cv_rgb = cv2.imread(os.path.join(test_data_path, img_file_name), cv2.IMREAD_COLOR)

        # # apply mask if available
        # if has_masks:
        #     mask = cv2.imread(os.path.join(test_data_path, 'masks', img_file_name+'.png'),0)
        #     cv_rgb = cv2.bitwise_and(cv_rgb,cv_rgb,mask = mask)

        # The bounding box
        # Must be square due to get_bbox2patch in imgproc.py
        img_height, img_width, _ = cv_rgb.shape
        # bounding_box = {
        #     "x_offset": 54,
        #     "y_offset": 120,
        #     "width": 350, #img_width
        #     "height": 350 #img_height
        # }
        bounding_box = {
            "x_offset": 1,
            "y_offset": 1,
            "width": 700, #img_width
            "height": 700 #img_height
        }
        # Build the request
        request = {
            "rgb_image": cv_rgb,
            "bounding_box": bounding_box
        }
        response = detect_keypoint.handle_keypoint_request(request)
        # print(response)

        for i in range(response["num_keypoints"]):
            cv_rgb = cv2.circle(cv_rgb, np.round(response["keypoints_pixels"][i]).astype(np.int32), radius=6, color=(0, 0, 255), thickness=2)
        cv_rgb = cv2.rectangle(cv_rgb, [bounding_box["x_offset"], bounding_box["y_offset"]],
                                [bounding_box["x_offset"] + bounding_box["width"],
                                 bounding_box["y_offset"] + bounding_box["height"]], color = (0, 255, 255), thickness = 1)
        if len(video_save_path) > 0:
            video.write(cv_rgb)
        elif visualize:
            cv2.imshow('Press any key to quit',cv_rgb)
            cv2.waitKey(0) # waits until a key is pressed
            cv2.destroyAllWindows() # destroys the window showing image
        if len(imgs_save_path) > 0:
            cv2.imwrite(os.path.join(imgs_save_path, videoname + '_' + str(img_count) + '.png'), cv_rgb)
    end_time = perf_counter()
    print("time per frame: ", (end_time - start_time)/len(os.listdir(test_data_path))*1000, " ms" )
    if len(video_save_path) > 0:
        video.release()
        print('Video saved to ', video_full_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--visualize', '-v', type=int,
        default=0)
    args = parser.parse_args()
    visualize = args.visualize
    main(visualize)
