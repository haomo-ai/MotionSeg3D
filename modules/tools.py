#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt


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


# def one_hot_pred_from_label(y_pred, labels):
#     y_true = torch.zeros_like(y_pred)
#     ones = torch.ones_like(y_pred)
#     indexes = [l for l in labels]
#     y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(
#         labels.size(0)), indexes]
#     return y_true


# def keep_variance_fn(x):
#     return x + 1e-3


# class SoftmaxHeteroscedasticLoss(torch.nn.Module):
#     def __init__(self):
#         super(SoftmaxHeteroscedasticLoss, self).__init__()
#         self.adf_softmax = adf.Softmax(
#             dim=1, keep_variance_fn=keep_variance_fn)

#     def forward(self, outputs, targets, eps=1e-5):
#         mean, var = self.adf_softmax(*outputs)
#         targets = torch.nn.functional.one_hot(
#             targets, num_classes=20).permute(0, 3, 1, 2).float()

#         precision = 1 / (var + eps)
#         return torch.mean(0.5 * precision * (targets - mean) ** 2 + 0.5 * torch.log(var + eps))


def save_to_txtlog(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return


def save_checkpoint(to_save, logdir, suffix=""):
    # Save the weights
    torch.save(to_save, logdir +
               "/SalsaNextWithMotionAttention" + suffix)


def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)


def make_log_img(depth, mask, pred, gt, color_fn):
    # input should be [depth, pred, gt]
    # make range image (normalized to 0,1 for saving)
    depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX,
                           dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
    out_img = cv2.applyColorMap(
        depth, get_mpl_colormap('viridis')) * mask[..., None]
    # make label prediction
    pred_color = color_fn((pred * mask).astype(np.int32))
    out_img = np.concatenate([out_img, pred_color], axis=0)
    # make label gt
    gt_color = color_fn(gt)
    out_img = np.concatenate([out_img, gt_color], axis=0)
    return (out_img).astype(np.uint8)


def show_scans_in_training(proj_mask, in_vol, argmax, proj_labels, color_fn):
    # get the first scan in batch and project points
    mask_np = proj_mask[0].cpu().numpy()
    depth_np = in_vol[0][0].cpu().numpy()
    pred_np = argmax[0].cpu().numpy()
    gt_np = proj_labels[0].cpu().numpy()
    out = make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)

    mask_np = proj_mask[1].cpu().numpy()
    depth_np = in_vol[1][0].cpu().numpy()
    pred_np = argmax[1].cpu().numpy()
    gt_np = proj_labels[1].cpu().numpy()
    out2 = make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)

    out = np.concatenate([out, out2], axis=0)

    cv2.imshow("sample_training", out)
    cv2.waitKey(1)


class iouEval:
    def __init__(self, n_classes, device, ignore=None):
        self.n_classes = n_classes
        self.device = device
        # if ignore is larger than n_classes, consider no ignoreIndex
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]).long()
        print("[IOU EVAL] IGNORE: ", self.ignore)
        print("[IOU EVAL] INCLUDE: ", self.include)
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.n_classes, self.n_classes), device=self.device).long()
        self.ones = None
        self.last_scan_size = None  # for when variable scan size is used

    def addBatch(self, x, y):  # x=preds, y=targets
        # if numpy, pass to pytorch
        # to tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().to(self.device)

        # sizes should be "batch_size x H x W"
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # idxs are labels and predictions
        idxs = torch.stack([x_row, y_row], dim=0)

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (cols = gt, rows = pred)
        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), self.ones, accumulate=True)

    def getStats(self):
        # remove fp and fn from confusion on the ignore classes cols and rows
        conf = self.conf_matrix.clone().double()
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"
