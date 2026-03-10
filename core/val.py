# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt


def val_net(cfg, epoch_idx, writer, data_loader, encoder, pointNetEncoder, fusion, decoder, refiner):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    with open(cfg.DATASETS[cfg.DATASET.TRAIN_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Validating loop
    n_samples = len(data_loader)
    val_iou = dict()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    pointNetEncoder.eval()
    fusion.eval()
    decoder.eval()
    refiner.eval()

    with tqdm(data_loader, desc='validate network') as data_loader:
        for sample_idx, (taxonomy_ids, taxonomy_names, sample_names, rendering_images, ground_truth_volumes, image_points) in enumerate(data_loader):
            taxonomy_id = taxonomy_ids[0]
            sample_name = sample_names[0]
            ground_truth_volume = ground_truth_volumes[0]

            with torch.no_grad():
                # Get data from data loader
                rendering_image = utils.network_utils.var_or_cuda(rendering_images)
                image_points = utils.network_utils.var_or_cuda(image_points)
                ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

                image_features = encoder(rendering_image)
                batch_size, n_views, channels, f_h, f_w = image_features.size()
                image_features = image_features.view(batch_size, channels, f_h, f_w).contiguous()
                point_features = pointNetEncoder(image_points)
                point_features = point_features.view(batch_size, channels, f_h, f_w).contiguous()
                #fused_features = torch.cat([image_features, point_features], 1)  # 拼接两种模态的特征
                fused_features = fusion(image_features, point_features)
                raw_features, generated_volume = decoder(fused_features)
                generated_volume = generated_volume.squeeze()

                unoccupied_voxels = torch.eq(ground_truth_volume, 0)
                occupied_voxels = torch.eq(ground_truth_volume, 1)
                Fpos_ce = bce_loss(generated_volume[unoccupied_voxels], ground_truth_volume[unoccupied_voxels])
                Fneg_ce = bce_loss(generated_volume[occupied_voxels], ground_truth_volume[occupied_voxels])
                encoder_loss = (torch.pow(Fpos_ce, 2) + torch.pow(Fneg_ce, 2))

                generated_volume = refiner(generated_volume)
                generated_volume = generated_volume.squeeze()
                unoccupied_voxels = torch.eq(ground_truth_volume, 0)
                occupied_voxels = torch.eq(ground_truth_volume, 1)
                Fpos_ce = bce_loss(generated_volume[unoccupied_voxels], ground_truth_volume[unoccupied_voxels])
                Fneg_ce = bce_loss(generated_volume[occupied_voxels], ground_truth_volume[occupied_voxels])
                refiner_loss = (torch.pow(Fpos_ce, 2) + torch.pow(Fneg_ce, 2))

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
                if taxonomy_id not in val_iou:
                    val_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
                val_iou[taxonomy_id]['n_samples'] += 1
                val_iou[taxonomy_id]['iou'].append(sample_iou)

                '''# Print sample loss and IoU
                print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
                      (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
                       refiner_loss.item(), ['%.4f' % si for si in sample_iou]))'''

    # Output validating results
    mean_iou = []
    for taxonomy_id in val_iou:
        val_iou[taxonomy_id]['iou'] = np.mean(val_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(val_iou[taxonomy_id]['iou'] * val_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ VALIDATE RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in val_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%-7d' % val_iou[taxonomy_id]['n_samples'], end='\t')

        for ti in val_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall\t\t%-7d' % n_samples, end='\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add validating results to TensorBoard
    max_iou = np.max(mean_iou)
    if writer is not None:
        writer.add_scalar('Val/encoder_loss', encoder_losses.avg, epoch_idx)
        writer.add_scalar('Val/refiner_loss', refiner_losses.avg, epoch_idx)
        writer.add_scalar('Val/taxonomy_max_IoU', max_iou, epoch_idx)

    return max_iou
