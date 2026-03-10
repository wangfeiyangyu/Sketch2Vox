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
from models.ours.vggEncoder import vggEncoder
from models.ours.pointnetEncoder import PointNetEncoder
from models.ours.decoder import Decoder
from models.ours.transformer import Transformer
from models.ours.refiner import Refiner

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
    #数据增强
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Resize(IMG_SIZE),
        # utils.data_transforms.Binarize(threshold=0.2),
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
    transformer = Transformer(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        pointNetEncoder = torch.nn.DataParallel(pointNetEncoder).cuda()
        transformer = torch.nn.DataParallel(transformer).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()

    print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
    #预训练权重加载
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    epoch_idx = checkpoint['epoch_idx']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    pointNetEncoder.load_state_dict(checkpoint['pointNetEncoder_state_dict'])
    transformer.load_state_dict(checkpoint['transformer_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    refiner.load_state_dict(checkpoint['refiner_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Validating loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    pointNetEncoder.eval()
    transformer.eval()
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

                # Validate the encoder, decoder, refiner and merger
                image_features = encoder(rendering_image)
                batch_size, n_views, channels, f_h, f_w = image_features.size()
                image_features = image_features.view(batch_size, channels, f_h * f_w).contiguous()
                point_features = pointNetEncoder(image_point)
                features = torch.cat([image_features, point_features], 1)  # 拼接两种模态的特征
                features = transformer(features)
                raw_features, generated_volume = decoder(features)
                generated_volume = generated_volume.squeeze(dim=1)

                # encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10
                unoccupied_voxels = torch.eq(ground_truth_volume, 0)
                occupied_voxels = torch.eq(ground_truth_volume, 1)
                Fpos_ce = bce_loss(generated_volume[unoccupied_voxels], ground_truth_volume[unoccupied_voxels])
                Fneg_ce = bce_loss(generated_volume[occupied_voxels], ground_truth_volume[occupied_voxels])
                encoder_loss = (torch.pow(Fpos_ce, 2) + torch.pow(Fneg_ce, 2)) * 10

                generated_volume = refiner(generated_volume)
                unoccupied_voxels = torch.eq(ground_truth_volume, 0)
                occupied_voxels = torch.eq(ground_truth_volume, 1)
                Fpos_ce = bce_loss(generated_volume[unoccupied_voxels], ground_truth_volume[unoccupied_voxels])
                Fneg_ce = bce_loss(generated_volume[occupied_voxels], ground_truth_volume[occupied_voxels])
                refiner_loss = (torch.pow(Fpos_ce, 2) + torch.pow(Fneg_ce, 2)) * 10

                # Append loss and accuracy to average metrics
                encoder_losses.update(encoder_loss.item())
                refiner_losses.update(refiner_loss.item())

                # IoU per sample
                sample_iou = []
                for th in cfg.TEST.VOXEL_THRESH:
                    _volume = torch.ge(generated_volume, th).float()
                    intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                    union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                    sample_iou.append((intersection / union).item())

                # IoU per taxonomy
                if taxonomy_id not in test_iou:
                    test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
                test_iou[taxonomy_id]['n_samples'] += 1
                test_iou[taxonomy_id]['iou'].append(sample_iou)

                '''# Append generated volumes
                if output_dir:
                    img_dir = output_dir % 'images'
                    # Volume Visualization
                    gv = generated_volume.cpu().numpy()
                    rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'best-epoch-'+str(epoch_idx)), sample_name, max(sample_iou), epoch_idx)
                    gtv = ground_truth_volume.cpu().numpy()
                    rendering_views = utils.binvox_visualization.get_volume_views(gtv, os.path.join(img_dir, 'best-epoch-'+str(epoch_idx)), sample_name+'-gt', max(sample_iou), epoch_idx)'''

                '''# Print sample loss and IoU
                print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
                      (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
                       refiner_loss.item(), ['%.4f' % si for si in sample_iou]))'''

    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
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
