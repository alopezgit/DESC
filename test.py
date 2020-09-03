# The code to evaluate the predictions given below is adapted from
# https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py
from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import time
import torch.nn
from options.test_options import TestOptions
from data import create_dataloader
from models import create_model
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import DataLoader


# This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
cv2.setNumThreads(0)


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate_predictions(pred_depths, gt_depths, crop_eigen=1, disable_median_scaling=1, max_depth=80, min_depth=1e-3):
    """Evaluates the given predictions with the gt_depths
    """

    MIN_DEPTH = min_depth
    MAX_DEPTH = max_depth

    errors = []
    ratios = []

    for i in range(len(pred_depths)):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

        if crop_eigen:
            # We evaluate in the paper mainly in the eigen split,
            # which uses this crop
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        if not disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(
            med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                           "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    list_results = []
    list_results.append(("{:>8} | " * 7).format("abs_rel",
                                                "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + "\n  ")
    list_results.append(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()))
    return list_results, mean_errors


def evaluate_model(model, data_loader, gt_depths_load, disable_median_scaling=0):
    """ The code below takes a model and predicts the depth for the test images
    """
    model.eval()
    gt_depths = []
    pred_depths = []
    for ind, data in enumerate(data_loader):
        gt_depth = np.squeeze(data['depth'].data.numpy())
        w = gt_depth.shape[1]
        h = gt_depth.shape[0]
        data['depth'] = torch.from_numpy(gt_depths_load[ind]).unsqueeze(0)
        with torch.no_grad():
            model.set_input(data)
            model.test()

        pred_depth = np.squeeze(model.pred.data.cpu().numpy())
        w0 = pred_depth.shape[1]

        # The predicted depth is upscaled to the original gt size
        pred_depth = cv2.resize(pred_depth, (w, h), cv2.INTER_CUBIC)

        # Our model predicts in the range [-1, 1], so we transform
        # the range to [0, max_depth]
        pred_depth += 1.0
        pred_depth /= 2.0
        pred_depth *= model.max_depth
        pred_depth[pred_depth < 1e-3] = 1e-3
        gt_depths.append(gt_depths_load[ind])
        pred_depths.append(pred_depth)

    # Following T2Net paper, we evaluate using cap 0-80m and 1-50m
    print('-'*40+'\n Evaluating Cap 80m\n'+'-'*40)
    results_80, mean_errors_80 = evaluate_predictions(
        pred_depths, gt_depths, crop_eigen=1, disable_median_scaling=disable_median_scaling, max_depth=80)
    print(results_80)
    print('-'*40+'\n Evaluating 1-50m\n'+'-'*40)
    results_50, mean_errors_50 = evaluate_predictions(
        pred_depths, gt_depths, crop_eigen=1, disable_median_scaling=disable_median_scaling, max_depth=50, min_depth=1)
    print(results_50)
    return mean_errors_80, mean_errors_50


if __name__ == '__main__':
    # The ground-truth is prepared using export_gt_depth.py
    # in https://github.com/nianticlabs/monodepth2
    gt_depths_load = np.load(
        './gt_depths.npz', fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    opt = TestOptions().parse()
    opt.root = opt.root.strip()
    test_data_loader = create_dataloader(opt)
    model = create_model(opt)
    model.setup(opt)
    model.load_networks('latest_joint_training')

    # In case you want to load the pretrained models you can use
    # the following lines. Specify where the model is saved via
    # the load dir argument

    # To load our main trained network in the paper
    # model.load_networks('DESC', load_dir='./pretrained_models/')

    # To load our model trained also with stereo photometric supervision
    # model.load_networks('DESC_stereo', load_dir='./pretrained_models/')
    # Also, if you want to evaluate the stereo-trained model, you need to set
    # the disable_median_scaling flag to 1 in the evaluate_model function

    evaluate_model(model, test_data_loader, gt_depths_load,
                   disable_median_scaling=0)
