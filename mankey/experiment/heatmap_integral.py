import torch
import os
import random
from torch.utils.data import DataLoader
import sys, os
mankey_path = os.path.dirname(os.path.dirname(sys.path[0]))
print('mankey_path: ', mankey_path)
sys.path.append(mankey_path)
from mankey.network.resnet_nostage import ResnetNoStageConfig, ResnetNoStage, init_from_modelzoo
from mankey.network.weighted_loss import weighted_mse_loss, weighted_l1_loss
import mankey.network.predict as predict
import mankey.network.visualize_dbg as visualize_dbg
import mankey.config.parameter as parameter
from mankey.dataproc.spartan_supervised_db import SpartanSupvervisedKeypointDBConfig, SpartanSupervisedKeypointDatabase
from mankey.dataproc.supervised_keypoint_loader import SupervisedKeypointDatasetConfig, SupervisedKeypointDataset


# Some global parameter
learning_rate = 2e-4
n_epoch = 121

def construct_dataset(is_train: bool) -> (SupervisedKeypointDataset, SupervisedKeypointDatasetConfig):
    # Construct the db info
    db_config = SpartanSupvervisedKeypointDBConfig()
    db_config.keypoint_yaml_name = 'mug_3_keypoint_image.yaml'
    db_config.pdc_data_root = '/home/ANT.AMAZON.COM/yifanhou/git/manip_dataset/data'
    if is_train:
        db_config.config_file_path = '/home/ANT.AMAZON.COM/yifanhou/git/mankey-ros/mankey/config/mugs_up_with_flat_logs.txt'
    else:
        db_config.config_file_path = '/home/ANT.AMAZON.COM/yifanhou/git/mankey-ros/mankey/config/mugs_up_with_flat_test_logs.txt'

    # Construct the database
    print(db_config)
    database = SpartanSupervisedKeypointDatabase(db_config)
    print('database.num_keypoints: ', database.num_keypoints)
    # Construct torch dataset
    config = SupervisedKeypointDatasetConfig()
    config.network_in_patch_width = 256
    config.network_in_patch_height = 256
    config.network_out_map_width = 64
    config.network_out_map_height = 64
    config.image_database_list.append(database)
    config.is_train = is_train
    dataset = SupervisedKeypointDataset(config)
    return dataset, config


def construct_network():
    net_config = ResnetNoStageConfig()
    net_config.num_keypoints = 3
    net_config.image_channels = 3
    net_config.depth_per_keypoint = 1
    net_config.num_layers = 34
    network = ResnetNoStage(net_config)
    return network, net_config


def visualize(network_path: str, save_dir: str):
    # Get the network
    network, _ = construct_network()

    # Load the network
    network.load_state_dict(torch.load(network_path))
    network.cuda()
    network.eval()

    # Construct the dataset
    dataset, config = construct_dataset(is_train=False)

    # try the entry
    num_entry = 50
    entry_idx = []
    for i in range(num_entry):
        entry_idx.append(random.randint(0, len(dataset) - 1))

    # A good example and a bad one
    for i in range(len(entry_idx)):
        visualize_dbg.visualize_entry_nostage(entry_idx[i], network, dataset, config, save_dir)


def train(checkpoint_dir: str, start_from_ckpnt: str = '', save_epoch_offset: int = 0):
    # Construct the dataset
    dataset_train, train_config = construct_dataset(is_train=True)
    # dataset_val, val_config = construct_dataset(is_train=False)

    # And the dataloader
    loader_train = DataLoader(dataset=dataset_train, batch_size=16, shuffle=True, num_workers=8)
    # loader_val = DataLoader(dataset=dataset_val, batch_size=16, shuffle=False, num_workers=4)

    # Construct the regressor
    network, net_config = construct_network()
    if len(start_from_ckpnt) > 0:
        network.load_state_dict(torch.load(start_from_ckpnt))
    else:
        init_from_modelzoo(network, net_config)
    network.cuda()

    # The checkpoint
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # The optimizer and scheduler
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 90], gamma=0.1)

    # The training loop
    for epoch in range(n_epoch):
        # Save the network
        if epoch % 4 == 0 and epoch > 0:
            file_name = 'checkpoint-%d.pth' % (epoch + save_epoch_offset)
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            print('Save the network at %s' % checkpoint_path)
            torch.save(network.state_dict(), checkpoint_path)

        # Prepare info for training
        network.train()
        train_error_xy = 0
        train_error_depth = 0

        # The learning rate step
        scheduler.step()
        for param_group in optimizer.param_groups:
            print('The learning rate is ', param_group['lr'])

        # The training iteration over the dataset
        for idx, data in enumerate(loader_train):
            # Get the data
            image = data[parameter.rgb_image_key]
            keypoint_xy = data[parameter.keypoint_xy_key]
            keypoint_weight = data[parameter.keypoint_validity_key]

            # Upload to GPU
            image = image.cuda()
            keypoint_xy = keypoint_xy.cuda()
            keypoint_weight = keypoint_weight.cuda()

            # To predict
            optimizer.zero_grad()
            raw_pred = network(image)
            prob_pred = raw_pred[:, 0:net_config.num_keypoints, :, :]
            # depthmap_pred = raw_pred[:, net_config.num_keypoints:, :, :]
            heatmap = predict.heatmap_from_predict(prob_pred, net_config.num_keypoints)
            _, _, heatmap_height, heatmap_width = heatmap.shape

            # Compute the coordinate
            coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap, net_config.num_keypoints)

            # Concantate them
            xy_pred = torch.cat((coord_x, coord_y), dim=2)

            # Compute loss
            loss = weighted_l1_loss(xy_pred, keypoint_xy, keypoint_weight)
            loss.backward()
            optimizer.step()

            # cleanup
            del loss

            # Log info
            xy_error = float(weighted_l1_loss(xy_pred[:, :, 0:2], keypoint_xy[:, :, 0:2], keypoint_weight[:, :, 0:2]).item())
            if idx % 100 == 0:
                print('Iteration %d in epoch %d' % (idx, epoch))
                print('The averaged pixel error is (pixel in 256x256 image): ', 256 * xy_error / len(xy_pred))

            # Update info
            train_error_xy += float(xy_error)

        # The info at epoch level
        print('Epoch %d' % epoch)
        print('The training averaged pixel error is (pixel in 256x256 image): ', 256 * train_error_xy / len(dataset_train))

if __name__ == '__main__':
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'ckpnt')

    # train
    train(checkpoint_dir=checkpoint_dir)

    # # visualization
    # net_path = os.path.join(checkpoint_dir, 'checkpoint-116.pth')
    # tmp_dir = 'tmp'
    # if not os.path.exists(tmp_dir):
    #    os.mkdir(tmp_dir)
    # visualize(net_path, tmp_dir)
