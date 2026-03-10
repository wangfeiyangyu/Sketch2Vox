# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt


def val_net(cfg, epoch_idx=-1, writer=None, output_dir=None, val_data_loader=None, encoder=None, decoder=None, refiner=None, merger=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(val_data_loader)
    val_iou = dict()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    for batch_idx, (taxonomy_ids, taxonomy_names, sample_names, rendering_images, ground_truth_volume, image_point) in enumerate(val_data_loader):
        taxonomy_id = taxonomy_ids[0]
        #sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            raw_features, generated_volume = decoder(image_features)
            generated_volume = merger(raw_features, generated_volume)
            encoder_loss = bce_loss(generated_volume, ground_truth_volume)
            generated_volume = refiner(generated_volume)
            refiner_loss = bce_loss(generated_volume, ground_truth_volume)

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


    # Output testing results
    mean_iou = []
    for taxonomy_id in val_iou:
        val_iou[taxonomy_id]['iou'] = np.mean(val_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(val_iou[taxonomy_id]['iou'] * val_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ Val RESULTS ============================')
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

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)

    if writer is not None:
        writer.add_scalar('Val/encoder_loss', encoder_losses.avg, epoch_idx)
        writer.add_scalar('Val/refiner_loss', refiner_losses.avg, epoch_idx)
        writer.add_scalar('Val/Max_IoU', max_iou, epoch_idx)

    return max_iou
