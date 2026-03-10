# -*- coding: utf-8 -*-
import os
import random
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.PointNet_Merger.test_Merger import test_net
from models.encoder import Encoder
from models.pointnet.pointnetEncoder import PointNetEncoder
from models.decoder import Decoder
from models.merger import Merger
from models.refiner import Refiner


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
        #utils.data_transforms.Binarize(threshold=0.2),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.TRAIN.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  num_workers=1,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up networks
    encoder = Encoder(cfg)
    pointNetEncoder = PointNetEncoder(cfg)
    pointNetDecoder = Decoder(cfg)
    decoder = Decoder(cfg)
    merger = Merger(cfg)
    refiner = Refiner(cfg)
    print('[DEBUG] %s Parameters in Encoder: %d.' % (dt.now(), utils.network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in PointNetEncoder: %d.' % (dt.now(), utils.network_utils.count_parameters(pointNetEncoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in PointNetDecoder: %d.' % (dt.now(), utils.network_utils.count_parameters(pointNetDecoder)))
    print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), utils.network_utils.count_parameters(merger)))
    print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(refiner)))

    # Initialize weights of networks
    encoder.apply(utils.network_utils.init_weights)
    pointNetEncoder.apply(utils.network_utils.init_weights)
    decoder.apply(utils.network_utils.init_weights)
    pointNetDecoder.apply(utils.network_utils.init_weights)
    merger.apply(utils.network_utils.init_weights)
    refiner.apply(utils.network_utils.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        pointNetEncoder_solver = torch.optim.Adam(pointNetEncoder.parameters(), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        pointNetDecoder_solver = torch.optim.Adam(pointNetDecoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        merger_solver = torch.optim.Adam(merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        refiner_solver = torch.optim.Adam(refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        pointNetEncoder_solver = torch.optim.SGD(pointNetEncoder.parameters(), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        pointNetDecoder_solver = torch.optim.SGD(pointNetDecoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        merger_solver = torch.optim.SGD(merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        refiner_solver = torch.optim.SGD(refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver, milestones=cfg.TRAIN.ENCODER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    pointNetEncoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(pointNetEncoder_solver, milestones=cfg.TRAIN.ENCODER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver, milestones=cfg.TRAIN.DECODER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    pointNetDecoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(pointNetDecoder_solver, milestones=cfg.TRAIN.DECODER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(merger_solver, milestones=cfg.TRAIN.MERGER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver, milestones=cfg.TRAIN.REFINER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        pointNetEncoder = torch.nn.DataParallel(pointNetEncoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        pointNetDecoder = torch.nn.DataParallel(pointNetDecoder).cuda()
        merger = torch.nn.DataParallel(merger).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    # mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
    # euclidean_loss = torch.nn.PairwiseDistance(p=2)

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat().replace(":" , "_"))
    log_dir = output_dir % 'logs'
    ckpt_dir = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    # Training finetune All
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES_FINETUNE):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        data_time = utils.network_utils.AverageMeter()
        encoder_losses = utils.network_utils.AverageMeter()
        refiner_losses = utils.network_utils.AverageMeter()

        # switch models to training mode
        encoder.train()
        pointNetEncoder.train()
        decoder.train()
        pointNetDecoder.train()
        merger.train()
        refiner.train()

        batch_end_time = time()
        for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_volumes, image_points) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)
            image_points = utils.network_utils.var_or_cuda(image_points)

            unoccupied_voxels = torch.eq(ground_truth_volumes, 0)
            occupied_voxels = torch.eq(ground_truth_volumes, 1)

            # Train the encoder, pointNetEncoder, decoder, pointNetDecoder, refiner
            image_features = encoder(rendering_images).squeeze(dim=1)
            point_features = pointNetEncoder(image_points)
            raw_feature, coarse_volumes = decoder(image_features)
            raw_feature_p, coarse_volumes_p = pointNetDecoder(point_features)
            raw_feature = torch.stack([raw_feature, raw_feature_p], dim=1)
            coarse_volumes = torch.cat([coarse_volumes, coarse_volumes_p], dim=1)
            generated_volumes = merger(raw_feature, coarse_volumes)

            encoder_Fpos_ce = bce_loss(generated_volumes[unoccupied_voxels], ground_truth_volumes[unoccupied_voxels])
            encoder_Fneg_ce = bce_loss(generated_volumes[occupied_voxels], ground_truth_volumes[occupied_voxels])
            encoder_loss = (torch.pow(encoder_Fpos_ce, 2) + torch.pow(encoder_Fneg_ce, 2)) * 10

            generated_volumes = refiner(generated_volumes)

            refiner_Fpos_ce = bce_loss(generated_volumes[unoccupied_voxels], ground_truth_volumes[unoccupied_voxels])
            refiner_Fneg_ce = bce_loss(generated_volumes[occupied_voxels], ground_truth_volumes[occupied_voxels])
            refiner_loss = (torch.pow(refiner_Fpos_ce, 2) + torch.pow(refiner_Fneg_ce, 2)) * 10

            # Gradient decent
            encoder.zero_grad()
            pointNetEncoder.zero_grad()
            decoder.zero_grad()
            pointNetDecoder.zero_grad()
            merger.zero_grad()
            refiner.zero_grad()

            encoder_loss.backward(retain_graph=True)
            refiner_loss.backward()

            encoder_solver.step()
            pointNetEncoder_solver.step()
            decoder_solver.step()
            pointNetDecoder_solver.step()
            refiner_solver.step()

        # Adjust learning rate
        encoder_lr_scheduler.step()
        pointNetEncoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        pointNetDecoder_lr_scheduler.step()
        merger_lr_scheduler.step()
        refiner_lr_scheduler.step()

        # Append loss to average metrics
        encoder_losses.update(encoder_loss.item())
        refiner_losses.update(refiner_loss.item())

        # Tick / tock
        epoch_end_time = time()
        print('[Train Finetune loss] %s Epoch [%d/%d] EpochTime = %.3f (s) encoder_loss = %.4f refiner_loss = %.4f'
              % (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES_FINETUNE, epoch_end_time - epoch_start_time, encoder_losses.avg, refiner_losses.avg))
        # Append epoch loss to TensorBoard
        #train_writer.add_scalar('Finetune/regression_loss', encoder_losses.avg, epoch_idx + 1)
        #train_writer.add_scalar('Finetune/reconstruction_loss', refiner_losses.avg, epoch_idx + 1)
        #train_writer.add_scalar('Finetune/finetune_loss', finetune_losses.avg, epoch_idx + 1)

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, pointNetEncoder, decoder, pointNetDecoder, merger, refiner)

        '''# Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg=cfg, file_path=os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1)), epoch_idx=epoch_idx + 1, encoder=encoder, encoder_solver=encoder_solver,
                                                 decoder=decoder,decoder_solver=decoder_solver, refiner=refiner,refiner_solver=refiner_solver, best_iou=best_iou, best_epoch=best_epoch)
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_iou = iou
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg=cfg, file_path=os.path.join(ckpt_dir, 'best-ckpt.pth'),epoch_idx=epoch_idx + 1, encoder=encoder,encoder_solver=encoder_solver, decoder=decoder,
                                                 decoder_solver=decoder_solver,refiner=refiner, refiner_solver=refiner_solver, best_iou=best_iou, best_epoch=best_epoch)'''

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()