# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from models.Ablation.no_attention.vggEncoder import vggEncoder
from models.Ablation.no_attention.pointnetEncoder import PointNetEncoder
from models.Ablation.no_attention.decoder import Decoder
from models.Ablation.no_attention.refiner import Refiner

def test_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat().replace(":" , "_"))

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TRAIN_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Resize(IMG_SIZE),
        #utils.data_transforms.Binarize(threshold=0.2),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                   batch_size=1,
                                                   num_workers=1,
                                                   pin_memory=True,
                                                   shuffle=False)

    # Set up networks
    encoder = vggEncoder(cfg)
    pointNetEncoder = PointNetEncoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        pointNetEncoder = torch.nn.DataParallel(pointNetEncoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()

    print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    epoch_idx = checkpoint['epoch_idx']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    print('[DEBUG] %s Loading Parameters in vggEncoder: %d.' % (dt.now(), utils.network_utils.count_parameters(encoder)))
    pointNetEncoder.load_state_dict(checkpoint['pointNetEncoder_state_dict'])
    print('[DEBUG] %s Loading Parameters in PointNetEncoder: %d.' % (dt.now(), utils.network_utils.count_parameters(pointNetEncoder)))
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    print('[DEBUG] %s Loading Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    refiner.load_state_dict(checkpoint['refiner_state_dict'])
    print('[DEBUG] %s Loading Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(refiner)))

    # Validating loop
    test_iou = dict()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    pointNetEncoder.eval()
    decoder.eval()
    refiner.eval()


    with tqdm(test_data_loader, desc='test network') as test_data_loader:
        for sample_idx, (taxonomy_ids, taxonomy_names, sample_names, rendering_images, ground_truth_volumes, image_points) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_ids[0]
            sample_name = sample_names[0]

            with torch.no_grad():
                # Get data from data loader
                rendering_image = utils.network_utils.var_or_cuda(rendering_images)
                image_point = utils.network_utils.var_or_cuda(image_points)
                ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volumes)

                # Test the encoder, pointNetEncoder, fusion, decoder
                image_features = encoder(rendering_image)
                batch_size, n_views, channels, f_h, f_w = image_features.size()
                image_features = image_features.view(batch_size, channels, f_h * f_w).contiguous()
                point_features = pointNetEncoder(image_point)
                fused_features = torch.cat([image_features, point_features], 1)  # 拼接两种模态的特征
                raw_features, generated_volume = decoder(fused_features)
                generated_volume = generated_volume.squeeze(dim=1)

                # Calculate encoder_loss
                encoder_loss = utils.network_utils.MSFCEL(generated_volume, ground_truth_volume)

                # Test the refiner
                generated_volume = refiner(generated_volume)
                # Calculate refiner_loss
                refiner_loss = utils.network_utils.MSFCEL(generated_volume, ground_truth_volume)

                # Append loss and accuracy to average metrics
                encoder_losses.update(encoder_loss.item())
                refiner_losses.update(refiner_loss.item())

                # IoU per sample
                sample_iou = []
                for th in cfg.TEST.VOXEL_THRESH:
                    _volume = torch.ge(generated_volume, th).float()
                    intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                    union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                    IoU = (intersection / union).item()
                    sample_iou.append(IoU)

                # IoU per taxonomy
                if taxonomy_id not in test_iou:
                    test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
                test_iou[taxonomy_id]['n_samples'] += 1
                test_iou[taxonomy_id]['iou'].append(sample_iou)

                '''# 保存测试生成的图像
                if output_dir:
                    img_dir = output_dir % 'images'
                    # Volume Visualization
                    gv = generated_volume.cpu().numpy()
                    rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'best-epoch-'+str(epoch_idx)), sample_name, max(sample_iou), epoch_idx)
                    gtv = ground_truth_volume.cpu().numpy()
                    rendering_views = utils.binvox_visualization.get_volume_views(gtv, os.path.join(img_dir, 'best-epoch-'+str(epoch_idx)), sample_name+'-gt', max(sample_iou), epoch_idx)'''

                # 保存测试生成的3D模型
                if output_dir:
                    binvox_dir = output_dir % 'binvoxs'
                    binvox_file = os.path.join(binvox_dir, taxonomy_id, '%s_%s_%.4f.binvox' % (sample_name, 'noAtten', max(sample_iou)))
                    #binvox_file = os.path.join(binvox_dir, taxonomy_id, sample_name, '%s.binvox' % (sample_name))
                    if not os.path.exists(os.path.dirname(binvox_file)):
                        os.makedirs(os.path.dirname(binvox_file))
                    gv = generated_volume.cpu().numpy()
                    gv = (gv.squeeze() > cfg.TEST.VOXEL_THRESH[0]).astype(np.int32)  # change numpy datatype to bool
                    with open(binvox_file, 'wb') as f:
                        voxel_model = utils.binvox_rw.Voxels(data=gv, dims=gv.shape, translate=[0, 0, 0], scale=1, axis_order='xyz')
                        voxel_model.write(f)

    # Output testing results
    mean_iou = []
    n_samples = 0  # len(test_data_loader)
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
        n_samples += test_iou[taxonomy_id]['n_samples']
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS (best epoch %d)============================' % epoch_idx)
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%-7d' % test_iou[taxonomy_id]['n_samples'], end='\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall\t\t%-7d' % n_samples, end='\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    max_iou = np.max(mean_iou)
    return max_iou
