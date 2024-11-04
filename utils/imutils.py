import random
import numpy as np
from PIL import Image
# from scipy import misc
import torch
import torchvision
from PIL import ImageEnhance


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    imgarr = np.asarray(img)
    proc_img = np.empty_like(imgarr, np.float32)

    proc_img[..., 0] = (imgarr[..., 0] - mean[0]) / std[0]
    proc_img[..., 1] = (imgarr[..., 1] - mean[1]) / std[1]
    proc_img[..., 2] = (imgarr[..., 2] - mean[2]) / std[2]
    return proc_img



def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label


def random_fliplr_multicd(pre_img, post_img, pre_lc_label, post_lc_label, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)
        pre_lc_label = np.fliplr(pre_lc_label)
        post_lc_label = np.fliplr(post_lc_label)

    return pre_img, post_img, pre_lc_label, post_lc_label, label

def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label


def random_flipud_multicd(pre_img, post_img, pre_lc_label, post_lc_label, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)
        pre_lc_label = np.flipud(pre_lc_label)
        post_lc_label = np.flipud(post_lc_label)
    return pre_img, post_img, pre_lc_label, post_lc_label, label



def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, label


def random_rot_multicd(pre_img, post_img, pre_lc_label, post_lc_label, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()
    pre_lc_label = np.rot90(pre_lc_label, k).copy()
    post_lc_label = np.rot90(post_lc_label, k).copy()

    return pre_img, post_img, pre_lc_label, post_lc_label, label



def random_crop(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    # pad_pre_image = np.zeros((H, W), dtype=np.float32)
    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    # pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w)] = pre_img
    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    # pre_img = pad_pre_image[H_start:H_end, W_start:W_end]
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
    # cmap = colormap()
    # misc.imsave('cropimg.png',image/255)
    # misc.imsave('croplabel.png',encode_cmap(label))
    return pre_img, post_img, label


def random_crop_multicd(pre_img, post_img, pre_lc_label, post_lc_label, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_pre_lc_label = np.zeros((H, W), dtype=np.float32)
    pad_post_lc_label = np.zeros((H, W), dtype=np.float32)

    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img

    pad_pre_lc_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = pre_lc_label
    pad_post_lc_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = post_lc_label
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    pre_lc_label = pad_pre_lc_label[H_start:H_end, W_start:W_end]
    post_lc_label = pad_post_lc_label[H_start:H_end, W_start:W_end]
    label = pad_label[H_start:H_end, W_start:W_end]
    # cmap = colormap()
    # misc.imsave('cropimg.png',image/255)
    # misc.imsave('croplabel.png',encode_cmap(label))
    return pre_img, post_img, pre_lc_label, post_lc_label, label

