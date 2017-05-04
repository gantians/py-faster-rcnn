# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def mask_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1]))
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1]] = im
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def prep_mask_for_blob(palette, im, boxes, mask_size):
    """Mean subtract and scale an image for use in a blob."""
    # boxes (x1, y1, x2, y2, cls)
    im = im.astype(np.float32, copy=False)
    mask = im.astype(np.float32, copy=False).reshape((-1, 3))
    mask = map(lambda x: palette[(int(x[0]), int(x[1]), int(x[2]))], mask[:])
    mask = np.array(mask).reshape(im.shape[0], im.shape[1])

    assert int(boxes[1]) <= int(boxes[3])
    assert int(boxes[0]) <= int(boxes[2])
    assert int(boxes[3]) <= im.shape[0]
    assert int(boxes[2]) <= im.shape[1]
    mask = mask[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])]
    mask[mask != boxes[4]] = 0
    mask[mask == boxes[4]] = 1
    mask_shape = mask.shape

    #print 'resized mask shape:{}'.format(mask.shape)
    im_scale_x = float(mask_size) / float(mask_shape[0])
    im_scale_y = float(mask_size) / float(mask_shape[1])

    mask = cv2.resize(mask, (mask_size, mask_size), None,
                    interpolation=cv2.INTER_NEAREST)
    return mask