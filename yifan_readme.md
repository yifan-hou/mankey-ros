Inference code:
nodes/mankey_keypoint_server.py:process_request_raw

Generate training data:
mankey/dataproc/scripts/pdc/keypoint_mesh2img_all_director.py 
need to provide all arguments.
The downloaded data seem to already have the training labels so no need to run this.

# Run training:
Before running, edit: 
1. db_config paths in the beginning
2. train or visualization in main
3. net_config_num_keypoints in parameter.py

python3 mankey/experiment/heatmap_integral.py

# Run inference:
python3 scripts/simple_mankey_client_test.py --visualize 1                                 


set netconfig:
heatmap_xxx.py
inference.py




* spartan_supervised_db.py
Database class that read the data files

* supervised_keypoint_db.py
Class for one data point entry

* dataproc/supervised_keypoint_loader.py:
dataSet class, which is a pytorch DataLoader, load entry from the constructed database