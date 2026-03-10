# -*- coding: utf-8 -*-
import os
import random
import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.Ablation.no_pointnet.val import val_net
from models.Ablation.no_pointnet.vggEncoder import vggEncoder
from models.Ablation.no_pointnet.decoder import Decoder
from models.Ablation.no_pointnet.refiner import Refiner


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Resize(IMG_SIZE),
        #utils.data_transforms.Binarize(threshold=0.2),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Resize(IMG_SIZE),
        # utils.data_transforms.Binarize(threshold=0.2),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.TRAIN.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  num_workers=1,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up networks
    encoder = vggEncoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    print('[DEBUG] %s Parameters in vggEncoder: %d.' % (dt.now(), utils.network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(refiner)))

    # Initialize weights of networks
    encoder.apply(utils.network_utils.init_weights)
    decoder.apply(utils.network_utils.init_weights)
    refiner.apply(utils.network_utils.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        refiner_solver = torch.optim.Adam(refiner.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    '''# Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver, milestones=cfg.TRAIN.ENCODER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    pointNetEncoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(pointNetEncoder_solver, milestones=cfg.TRAIN.ENCODER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    fusion_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(fusion_solver, milestones=cfg.TRAIN.MERGER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver, milestones=cfg.TRAIN.DECODER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver, milestones=cfg.TRAIN.REFINER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)'''

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()


    init_epoch = 0
    best_iou = -1
    best_epoch = -1

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat().replace(":" , "_"))
    ckpt_dir = output_dir % 'checkpoints'
    log_dir = output_dir % 'logs'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'val'))

    # Training
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        epoch_start_time = time()  # Tick/tock

        # Batch average meterics
        encoder_losses = utils.network_utils.AverageMeter()
        refiner_losses = utils.network_utils.AverageMeter()

        # switch models to training mode
        encoder.train()
        decoder.train()
        refiner.train()

        batch_end_time = time()
        with tqdm(train_data_loader, desc='train network') as train_data_loader:
            for batch_idx, (taxonomy_ids, taxonomy_names, sample_names, rendering_images, ground_truth_volumes, image_points) in enumerate(train_data_loader):
                # data_time.update(time() - batch_end_time)  # Measure data time

                # Get data from data loader
                rendering_images = utils.network_utils.var_or_cuda(rendering_images)
                ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)
                #image_points = utils.network_utils.var_or_cuda(image_points)

                unoccupied_voxels = torch.eq(ground_truth_volumes, 0)
                occupied_voxels = torch.eq(ground_truth_volumes, 1)

                # Train the encoder, decoder
                image_features = encoder(rendering_images)
                batch_size, n_views, channels, f_h, f_w = image_features.size()
                image_features = image_features.view(batch_size, channels, f_h*f_w).contiguous()

                raw_feature, generated_volumes = decoder(image_features)
                generated_volumes = generated_volumes.squeeze(dim=1)

                # Calculate encoder_loss
                encoder_loss = utils.network_utils.MSFCEL(generated_volumes, ground_truth_volumes)

                # Train the refiner
                generated_volumes = refiner(generated_volumes)
                # Calculate refiner_loss
                refiner_loss = utils.network_utils.MSFCEL(generated_volumes, ground_truth_volumes)

                # Gradient decent
                encoder.zero_grad()
                decoder.zero_grad()
                refiner.zero_grad()

                encoder_loss.backward(retain_graph=True)
                refiner_loss.backward()

                encoder_solver.step()
                decoder_solver.step()
                refiner_solver.step()

        '''# Adjust learning rate
        encoder_lr_scheduler.step()
        pointNetEncoder_lr_scheduler.step()
        transformer_lr_scheduler.step()
        decoder_lr_scheduler.step()
        refiner_lr_scheduler.step()'''

        # Append loss to average metrics
        encoder_losses.update(encoder_loss.item())
        refiner_losses.update(refiner_loss.item())

        # Tick / tock
        epoch_end_time = time()
        print('[Train loss] %s Epoch [%d/%d] EpochTime = %.3f (s) encoder_loss = %.4f refiner_loss = %.4f'
              % (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, encoder_losses.avg, refiner_losses.avg))
        # Append epoch loss to TensorBoard
        train_writer.add_scalar('Train/encoder_loss', encoder_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Train/refiner_loss', refiner_losses.avg, epoch_idx + 1)

        # Validate the training models
        iou = val_net(cfg=cfg, epoch_idx=epoch_idx+1, writer=val_writer, data_loader=val_data_loader,
                      encoder=encoder, decoder=decoder, refiner=refiner)

        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            best_iou = iou
            best_epoch = epoch_idx + 1
            ckpt_file = os.path.join(ckpt_dir, 'best-ckpt.pth')
            if os.path.exists(ckpt_file):
                os.remove(ckpt_file)
            utils.network_utils.save_checkpoints(cfg=cfg, file_path=ckpt_file, epoch_idx=epoch_idx + 1, best_epoch=best_epoch, best_iou=best_iou,
                                                 encoder=encoder, encoder_solver=encoder_solver,
                                                 decoder=decoder, decoder_solver=decoder_solver,
                                                 refiner=refiner, refiner_solver=refiner_solver)

    print("best_epoch: %s, best_iou: %s" % (best_epoch, best_iou))
    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()