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

from models.pointnet.pointnetEncoder import PointNetEncoder
from models.decoder import Decoder
from models.refiner import Refiner


def test_net(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             pointNetEncoder=None,
             decoder=None,
             pointNetDecoder=None,
             merger=None,
             refiner=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.Resize(IMG_SIZE),
            utils.data_transforms.Binarize(threshold=0.2),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=1,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_Fscore = dict()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    pointNetEncoder.eval()
    decoder.eval()
    pointNetDecoder.eval()
    merger.eval()
    refiner.eval()

    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume, image_points) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            image_points = utils.network_utils.var_or_cuda(image_points)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

            unoccupied_voxels = torch.eq(ground_truth_volume, 0)
            occupied_voxels = torch.eq(ground_truth_volume, 1)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images).squeeze(dim=1)
            point_features = pointNetEncoder(image_points)
            raw_feature, coarse_volume = decoder(image_features)
            raw_feature_p, coarse_volume_p = pointNetDecoder(point_features)
            raw_feature = torch.stack([raw_feature, raw_feature_p], dim=1)
            coarse_volume = torch.cat([coarse_volume, coarse_volume_p], dim=1)
            generated_volume = merger(raw_feature, coarse_volume)

            encoder_Fpos_ce = bce_loss(generated_volume[unoccupied_voxels], ground_truth_volume[unoccupied_voxels])
            encoder_Fneg_ce = bce_loss(generated_volume[occupied_voxels], ground_truth_volume[occupied_voxels])
            encoder_loss = (torch.pow(encoder_Fpos_ce, 2) + torch.pow(encoder_Fneg_ce, 2)) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refiner(generated_volume)
                Fpos_ce = bce_loss(generated_volume[unoccupied_voxels], ground_truth_volume[unoccupied_voxels])
                Fneg_ce = bce_loss(generated_volume[occupied_voxels], ground_truth_volume[occupied_voxels])
                refiner_loss = (torch.pow(Fpos_ce, 2) + torch.pow(Fneg_ce, 2)) * 10
            else:
                refiner_loss = encoder_loss

            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            # Fscore per sample
            sample_Fscore = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                # precession = intersection/torch.sum(_volume)
                # recall = intersection/torch.sum(ground_truth_volume)
                # F_score = 2*precession*recall / (precession+recall)
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_Fscore.append((intersection / union).item())
                # sample_Fscore.append(F_score.item())

            # Fscore per taxonomy
            if taxonomy_id not in test_Fscore:
                test_Fscore[taxonomy_id] = {'n_samples': 0, 'f_score': []}
            test_Fscore[taxonomy_id]['n_samples'] += 1
            test_Fscore[taxonomy_id]['f_score'].append(sample_Fscore)

            # Append generated volumes to TensorBoard
            if output_dir and sample_idx < 3:
                img_dir = output_dir % 'images'
                # Volume Visualization
                gv = generated_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'test'), epoch_idx)
                test_writer.add_image('Test Sample#%02d/Volume Reconstructed' % sample_idx, rendering_views, epoch_idx, dataformats='HWC')
                gtv = ground_truth_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_volume_views(gtv, os.path.join(img_dir, 'test'), epoch_idx)
                test_writer.add_image('Test Sample#%02d/Volume GroundTruth' % sample_idx, rendering_views, epoch_idx, dataformats='HWC')

            '''# Print sample loss and IoU
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
                  (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
                   refiner_loss.item(), ['%.4f' % si for si in sample_iou]))'''

    # Output testing results
    mean_Fscore = []
    for taxonomy_id in test_Fscore:
        test_Fscore[taxonomy_id]['f_score'] = np.mean(test_Fscore[taxonomy_id]['f_score'], axis=0)
        mean_Fscore.append(test_Fscore[taxonomy_id]['f_score'] * test_Fscore[taxonomy_id]['n_samples'])
    mean_Fscore = np.sum(mean_Fscore, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_Fscore:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_Fscore[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_Fscore[taxonomy_id]['f_score']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_Fscore:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_Fscore = np.max(mean_Fscore)
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/ENCODER_LOSS', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/REFINER_LOSS', refiner_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/TAXONOMY_MAX_Fscore', max_Fscore, epoch_idx)

    return max_Fscore
