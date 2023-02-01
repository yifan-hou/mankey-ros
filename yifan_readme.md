Inference code:
nodes/mankey_keypoint_server.py:process_request_raw

Generate training data:
mankey/dataproc/scripts/pdc/keypoint_mesh2img_all_director.py 
need to provide all arguments.
The downloaded data seem to already have the training labels so no need to run this.

Run training:
python3 mankey/experiment/heatmap_integral.py

Run inference:
python3 scripts/simple_mankey_client_test.py --visualize 1                                 



set netconfig:
heatmap_xxx.py
resnet_nostage.py
inference.py