# -*- coding: utf-8 -*-

import torch
from datetime import datetime as dt


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(cfg, file_path, epoch_idx, best_epoch, best_iou, encoder=None, encoder_solver=None, pointNetEncoder=None, pointNetEncoder_solver=None,
                     fusion=None, fusion_solver=None, decoder=None, decoder_solver=None, refiner=None, refiner_solver=None, merger=None, merger_solver=None):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))

    checkpoint = {
        'epoch_idx': epoch_idx,
        'best_iou': best_iou,
        'best_epoch': best_epoch}

    if encoder is not None:
        checkpoint['encoder_state_dict'] = encoder.state_dict()
    if encoder_solver is not None:
        checkpoint['encoder_solver_state_dict'] = encoder_solver.state_dict()
    if pointNetEncoder is not None:
        checkpoint['pointNetEncoder_state_dict'] = pointNetEncoder.state_dict()
    if pointNetEncoder_solver is not None:
        checkpoint['pointNetEncoder_solver_state_dict'] = pointNetEncoder_solver.state_dict()
    if fusion is not None:
        checkpoint['fusion_state_dict'] = fusion.state_dict()
    if fusion_solver is not None:
        checkpoint['fusion_solver_state_dict'] = fusion_solver.state_dict()
    if decoder is not None:
        checkpoint['decoder_state_dict'] = decoder.state_dict()
    if decoder_solver is not None:
        checkpoint['decoder_solver_state_dict'] = decoder_solver.state_dict()
    if refiner is not None:
        checkpoint['refiner_state_dict'] = refiner.state_dict()
    if refiner_solver is not None:
        checkpoint['refiner_solver_state_dict'] = refiner_solver.state_dict()
    if merger is not None:
        checkpoint['merger_state_dict'] = merger.state_dict()
    if merger_solver is not None:
        checkpoint['merger_solver_state_dict'] = merger_solver.state_dict()

    torch.save(checkpoint, file_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def MSFCEL(pred_value, ground_truth):
    """
        Mean Squared False Cross-Entropy Loss function
    @param ground_truth: ground truth volumes
    @param pred_value: generated_volumes
    @return: loss value in float
    """
    bce_loss = torch.nn.BCELoss()
    unoccupied_voxels = torch.eq(ground_truth, 0)
    occupied_voxels = torch.eq(ground_truth, 1)
    Fpos_ce = bce_loss(pred_value[unoccupied_voxels], ground_truth[unoccupied_voxels])
    Fneg_ce = bce_loss(pred_value[occupied_voxels], ground_truth[occupied_voxels])
    loss = (torch.pow(Fpos_ce, 2) + torch.pow(Fneg_ce, 2)) * 10
    return loss
