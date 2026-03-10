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

from core.Ablation.no_vgg.val import val_net
from models.Ablation.no_vgg.pointnetEncoder import PointNetEncoder
from models.Ablation.no_vgg.decoder import Decoder
from models.Ablation.no_vgg.refiner import Refiner


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
    pointNetEncoder = PointNetEncoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    print('[DEBUG] %s Parameters in PointNetEncoder: %d.' % (dt.now(), utils.network_utils.count_parameters(pointNetEncoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(refiner)))

    # Initialize weights of networks
    pointNetEncoder.apply(utils.network_utils.init_weights)
    decoder.apply(utils.network_utils.init_weights)
    refiner.apply(utils.network_utils.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        pointNetEncoder_solver = torch.optim.Adam(pointNetEncoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=cfg.TRAIN.BETAS)
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
        pointNetEncoder = torch.nn.DataParallel(pointNetEncoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    # mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
    # euclidean_loss = torch.nn.PairwiseDistance(p=2)

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
        # data_time = utils.network_utils.AverageMeter()
        encoder_losses = utils.network_utils.AverageMeter()
        refiner_losses = utils.network_utils.AverageMeter()

        # switch models to training mode
        pointNetEncoder.train()
        decoder.train()
        refiner.train()

        batch_end_time = time()
        with tqdm(train_data_loader, desc='train network') as train_data_loader:
            for batch_idx, (taxonomy_ids, taxonomy_names, sample_names, rendering_images, ground_truth_volumes, image_points) in enumerate(train_data_loader):
                # data_time.update(time() - batch_end_time)  # Measure data time

                # Get data from data loader
                #rendering_images = utils.network_utils.var_or_cuda(rendering_images)
                ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)
                image_points = utils.network_utils.var_or_cuda(image_points)

                unoccupied_voxels = torch.eq(ground_truth_volumes, 0)
                occupied_voxels = torch.eq(ground_truth_volumes, 1)

                # Train the pointNetEncoder, decoder, refiner
                point_features = pointNetEncoder(image_points)
                raw_feature, generated_volumes = decoder(point_features)

                generated_volumes = generated_volumes.squeeze(dim=1)
                # 计算encoder_loss
                encoder_Fpos_ce = bce_loss(generated_volumes[unoccupied_voxels], ground_truth_volumes[unoccupied_voxels])
                encoder_Fneg_ce = bce_loss(generated_volumes[occupied_voxels], ground_truth_volumes[occupied_voxels])
                encoder_loss = (torch.pow(encoder_Fpos_ce, 2) + torch.pow(encoder_Fneg_ce, 2)) * 10

                generated_volumes = refiner(generated_volumes)
                # 计算refiner_loss
                refiner_Fpos_ce = bce_loss(generated_volumes[unoccupied_voxels], ground_truth_volumes[unoccupied_voxels])
                refiner_Fneg_ce = bce_loss(generated_volumes[occupied_voxels], ground_truth_volumes[occupied_voxels])
                refiner_loss = (torch.pow(refiner_Fpos_ce, 2) + torch.pow(refiner_Fneg_ce, 2)) * 10

                # Gradient decent
                pointNetEncoder.zero_grad()
                decoder.zero_grad()
                refiner.zero_grad()

                encoder_loss.backward(retain_graph=True)
                refiner_loss.backward()

                pointNetEncoder_solver.step()
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
                      pointNetEncoder=pointNetEncoder, decoder=decoder, refiner=refiner)

        '''# Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg=cfg, file_path=os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1)), epoch_idx=epoch_idx + 1, encoder=encoder, encoder_solver=encoder_solver,
                                                 decoder=decoder,decoder_solver=decoder_solver, refiner=refiner,refiner_solver=refiner_solver, best_iou=best_iou, best_epoch=best_epoch)'''
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            best_iou = iou
            best_epoch = epoch_idx + 1
            ckpt_file = os.path.join(ckpt_dir, 'best-ckpt.pth')
            if os.path.exists(ckpt_file):
                os.remove(ckpt_file)
            utils.network_utils.save_checkpoints(cfg=cfg, file_path=ckpt_file, epoch_idx=epoch_idx + 1, best_epoch=best_epoch, best_iou=best_iou,
                                                 pointNetEncoder=pointNetEncoder, pointNetEncoder_solver=pointNetEncoder_solver,
                                                 decoder=decoder, decoder_solver=decoder_solver,
                                                 refiner=refiner, refiner_solver=refiner_solver)

    print("best_epoch: %s, best_iou: %s" % (best_epoch, best_iou))
    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()