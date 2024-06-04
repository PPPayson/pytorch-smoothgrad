import cv2
import numpy as np
import torch
from torch.autograd import Variable
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

def preprocess_image(img, cuda=False):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    if cuda:
        preprocessed_img = Variable(preprocessed_img.cuda(), requires_grad=True)
    else:
        preprocessed_img = Variable(preprocessed_img, requires_grad=True)

    return preprocessed_img


def save_as_gray_image(img, filename, percentile=99):
    img_2d = np.sum(img, axis=0)
    span = abs(np.percentile(img_2d, percentile))
    vmin = -span
    vmax = span
    img_2d = np.clip((img_2d - vmin) / (vmax - vmin), 0, 1)
    cv2.imwrite(filename, img_2d * 255)

    return

def save_as_heatmap(img, filename, percentile=99):
    img_2d = np.sum(img, axis=0)
    img_2d = gaussian_filter(img_2d, sigma=1)
    span = abs(np.percentile(img_2d, percentile))
    vmin = -span
    vmax = span
    img_2d = np.clip((img_2d-vmin)/(vmax-vmin)*2-1, -1, 1)
    #img_2d = np.clip((img_2d - vmin) / (vmax - vmin), -1, 1)
    ax = sns.heatmap(img_2d, cmap='viridis', xticklabels=False, yticklabels=False, vmin=-1, vmax=1)
    ax.figure.savefig(filename)
    plt.clf()
    return

def save_as_overlay(img, mask, filename, percentile=99):
    mask = np.sum(mask, axis=0)
    mask = gaussian_filter(mask, sigma=1)
    span = abs(np.percentile(mask, percentile))
    vmin = -span
    vmax = span
    mask = np.clip((mask-vmin)/(vmax-vmin)*2-1, -1, 1)
    alpha = np.ones((mask.shape[0], mask.shape[1]))
    thresh = abs(np.percentile(mask, 95))
    alpha[np.logical_and(mask>=0, mask<thresh)] = 0
    alpha[np.logical_and(mask<0, mask>-thresh)] = 0
    mask = (mask+1)/2
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2RGBA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    overlay = cv2.addWeighted(heatmap, 0.6, img, 0.4, 0)
    img[alpha!=0] = overlay[alpha!=0]
    cv2.imwrite(filename, img)
    return

def save_diff_map(img_1, img_2, org_img, filename_1, filename_2, percentile=99):
    img_1 = np.sum(img_1, axis=0)
    img_2 = np.sum(img_2, axis=0)
    span_1 = abs(np.percentile(img_1, percentile))
    vmin_1 = -span_1
    vmax_1 = span_1
    img_1 = np.clip((img_1-vmin_1)/(vmax_1-vmin_1)*2-1, -1, 1)
    
    span_2 = abs(np.percentile(img_2, percentile))
    vmin_2 = -span_2
    vmax_2 = span_2
    img_2 = np.clip((img_2-vmin_2)/(vmax_2-vmin_2)*2-1, -1, 1)
    
    diff = img_1 - img_2
    diff = gaussian_filter(diff, sigma=1)
    span = abs(np.percentile(diff, percentile))
    vmin = -span
    vmax = span
    diff = np.clip((diff-vmin)/(vmax-vmin)*2-1, -1, 1)
    alpha = np.ones((diff.shape[0], diff.shape[1]))
    thresh = abs(np.percentile(diff, 95))
    alpha[np.logical_and(diff>=0, diff<thresh)] = 0
    alpha[np.logical_and(diff<0, diff>-thresh)] = 0
    ax = sns.heatmap(diff, cmap='viridis', xticklabels=False, yticklabels=False, vmin=-1, vmax=1)
    ax.figure.savefig(filename_1)
    plt.clf()
    diff = (diff+1)/2
    heatmap = cv2.applyColorMap(np.uint8(255*diff), cv2.COLORMAP_VIRIDIS)
    overlay = cv2.addWeighted(heatmap, 0.6, org_img, 0.4, 0)
    org_img[alpha!=0] = overlay[alpha!=0]
    cv2.imwrite(filename_2, org_img)
    return
    
def save_cam_image(img, mask, filename):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_VIRIDIS)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(filename, np.uint8(255 * cam))
