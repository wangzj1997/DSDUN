"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional
import torch

import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)

from data.transforms import to_complex
import matplotlib.pyplot as plt

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval
    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]

def lfd(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate LFD (Log Frequency Distance).

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the LFD calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: lfd result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    img1 = img1.transpose(2, 0, 1)
    img2 = img2.transpose(2, 0, 1)
    freq1 = np.fft.fft2(img1)
    freq2 = np.fft.fft2(img2)
    return np.log(np.mean((freq1.real - freq2.real)**2 + (freq1.imag - freq2.imag)**2) + 1.0)

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    img = img.astype(np.float64)
    return img



METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    LFD=lfd,
)

class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """
    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        file = open("evaluate_result/result_{}.txt".format(args.name), 'w')
        for name in metric_names:
            file.write("{}: {:.4g} +/- {:.4g} \r\n".format(name,means[name],2 * stddevs[name]))
        file.close()
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )


def evaluate(args):
    metrics = Metrics(METRIC_FUNCS)
    for tgt_file in args.target_path.iterdir():
        tgt_kspace = np.load(tgt_file)  # K-space, (num_slice,256,256,2)
        tgt_kspace_tensor = torch.from_numpy(tgt_kspace)
        tgt_img = np.zeros(tgt_kspace_tensor.shape[:3])
        #print(tgt_img.shape)  # (num_slices,256,256)
        rec_img = np.load(args.predictions_path / tgt_file.name)   # image space, (num_slices,256,256)
        for i in range(len(tgt_kspace_tensor)):
            # tgt_img_tensor_i = torch.ifft(tgt_kspace_tensor[i,:,:,:], 2, normalized=True).type(torch.FloatTensor)
            # print(tgt_kspace_tensor.shape)
            tgt_img_tensor_i = torch.view_as_real(
                torch.fft.ifftn(  # type: ignore
                    torch.view_as_complex(tgt_kspace_tensor[i,:,:,:]), dim=(-2, -1), norm="ortho"
                )
            )
            tgt_img_i = tgt_img_tensor_i.numpy()
            tgt_img_i = np.abs(to_complex(tgt_img_i))
            # 每张图片范围压到 (0，255)
            tgt_img_i = (tgt_img_i-np.min(tgt_img_i))/(np.max(tgt_img_i)-np.min(tgt_img_i))*255
            tgt_img[i,:,:] = tgt_img_i
            
            # 每张图片范围压到 (0，255)
            rec_img[i,:,:] = (rec_img[i,:,:]-np.min(rec_img[i,:,:]))/(np.max(rec_img[i,:,:])-np.min(rec_img[i,:,:]))*255
        #print(rec_img.shape)  #(num_slices,256,256)

        '''
        print("tgt_img: ",np.min(tgt_img),np.max(tgt_img))
        print("rec_img: ",np.min(rec_img),np.max(rec_img))
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(tgt_img[50,:,:], cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(rec_img[50,:,:], cmap='gray')
        plt.show()
        '''

        tgt = tgt_img[args.i:,:,:]
        #rec = rec_img[args.i:,:,:]  #for CS and zerofilled
        rec = rec_img[:,:,:]   #for DL
        metrics.push(tgt_img, rec_img)
    return metrics


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--target-path",
        type=pathlib.Path,
        required=True,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "--predictions-path",
        type=pathlib.Path,
        required=True,
        help="Path to reconstructions",
    )

    parser.add_argument('--name',type=str,required = True,
        help = "name for evaluate results, eg : unet")
    parser.add_argument('--i',type=int,required = True)
    args = parser.parse_args()

    metrics = evaluate(args)
    print(metrics)
