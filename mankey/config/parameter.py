# The string key for dataset
# rgbd_image_key = 'rgbd_image'
rgb_image_key = 'rgbd_image'
# keypoint_xyd_key = 'normalized_keypoint_xyd'
keypoint_xy_key = 'normalized_keypoint_xy'
keypoint_validity_key = 'validity'
target_heatmap_key = 'target_heatmap'


net_config_num_keypoints = 2
net_config_image_channels = 3
net_config_depth_per_keypoint = 1
net_config_num_layers = 34

# The bounding box given by Database is tight, make it losser
bbox_scale = 1.25

# The normalization parameter
depth_image_clip = 2000  # Clip the depth image further than 1500 mm
depth_image_mean = 580
depth_image_scale = 256  # scaled_depth = (raw_depth - depth_image_mean) / depth_image_scale

# The averaged RGB image
rgb_mean = [0.485, 0.456, 0.406]

# The default size of path
default_patch_size_input = 256
default_patch_size_output = 64

# The camera parameter
focal_x = 533.6422696034836
focal_y = 534.7824445233571
principal_x = 319.4091030774892
principal_y = 236.4374299691866
