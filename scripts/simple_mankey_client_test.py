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
            # cv_depth,  # type: np.ndarray
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
    # rospy.wait_for_service('detect_keypoints')
    # detect_keypoint = rospy.ServiceProxy(
    #     'detect_keypoints', MankeyKeypointDetection)
    project_path = os.path.join(os.path.dirname(__file__), os.path.pardir)
    project_path = os.path.abspath(project_path)
    # Get the test data path
    test_data_path = os.path.join(project_path, 'test_data')
    cv_rbg_path = os.path.join(test_data_path, '000000_rgb.png')
    cv_depth_path = os.path.join(test_data_path, '000000_depth.png')

    # Read the image
    cv_rgb = cv2.imread(cv_rbg_path, cv2.IMREAD_COLOR)
    cv_depth = cv2.imread(cv_depth_path, cv2.IMREAD_ANYDEPTH)

    model_path = os.path.join(project_path, 'mankey/experiment/ckpnt/checkpoint-120.pth')
    detect_keypoint = MankeyKeypointDetectionServer(model_path)

    # The bounding box
    roi = {
        "x_offset": 261,
        "y_offset": 194,
        "width": 327 - 261,
        "height": 260 - 194
    }

    # Build the request
    # request = MankeyKeypointDetectionRequest()
    # bridge = CvBridge()
    request = {
        "rgb_image": cv_rgb,
        "depth_image": cv_depth,
        "bounding_box": roi
    }
    response = detect_keypoint.handle_keypoint_request(request)
    print(response)

    if visualize:
        for i in range(response["num_keypoints"]):
            cv_rgb = cv2.circle(cv_rgb, np.round(response["keypoints_pixels"][i]).astype(np.int32), radius=3, color=(255, 0, 0), thickness=1)

        cv2.imshow('Press any key to quit',cv_rgb)
        cv2.waitKey(0) # waits until a key is pressed
        cv2.destroyAllWindows() # destroys the window showing image


        # import open3d as o3d

        # vis_list = []

        # color = o3d.geometry.Image(cv_rgb)
        # depth = o3d.geometry.Image(cv_depth)
        # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)

        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        #     rgbd, o3d.camera.PinholeCameraIntrinsic(
        #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # vis_list.append(pcd)

        # for keypoint in response["keypoints_camera_frame"]:
        #     keypoints_coords \
        #         = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #             size=0.1, origin=keypoint)
        #     vis_list.append(keypoints_coords)
        # o3d.visualization.draw_geometries(vis_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--visualize', '-v', type=int,
        default=0)
    args = parser.parse_args()
    visualize = args.visualize
    main(visualize)
