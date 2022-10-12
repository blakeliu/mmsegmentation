import copy
import os.path as osp
import cv2
import mmcv
import numpy as np
import pytest
from mmcv.utils import build_from_cfg
from PIL import Image

from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines import Compose


def test_albu():
    # test assertion for invalid random crop
    # with pytest.raises(AssertionError):
    #     transform = dict(type='RandomCrop', crop_size=(-1, 0))
    #     build_from_cfg(transform, PIPELINES)

    results = dict()
    root = "/home/tf/data/disk/data/face/situ/wrinkles/seg"
    img = mmcv.imread(
        osp.join(root, 'images/val/people_84_Left_RGB.jpg'), 'color')
    seg = np.array(
        Image.open(osp.join(root, 'labels/val/people_84_Left_RGB.png')))
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    # albu.Sequential([albu.SmallestMaxSize(max_size=512, p=1.0),
    #                      albu.PadIfNeeded(min_height=1024, min_width=512, p=1.0, border_mode=cv2.BORDER_CONSTANT,
    #                                       value=(0, 0, 0)),

    #                      albu.OneOf([albu.RandomSizedCrop(min_max_height=(900, 1024), height=height, width=width,
    #                                                       w2h_ratio=0.5, p=0.5),
    #                                  albu.CenterCrop(height=1024, width=512, p=0.5), ], p=1.0)], p=0.5),

    h, w, _ = img.shape
    img_scale = (1024, 1280)  # w h
    crop_size = (1024, 1024)  # h w
    albu_transform = dict(type='Albu',
                          transforms=[
                                dict(type='Transpose', p=0.1),
                              dict(
                                  type='Sequential',
                                  transforms=[
                                      dict(type='SmallestMaxSize',
                                           max_size=max(img_scale), p=1),
                                      dict(type='PadIfNeeded', min_height=img_scale[1], min_width=img_scale[0], border_mode=0,
                                           value=(0, 0, 0), p=1),
                                  ]
                              ),
                              dict(
                                  type='OneOf',
                                  transforms=[
                                      dict(
                                          type='RandomSizedCrop',
                                          min_max_height=img_scale,
                                          height=crop_size[1],
                                          width=crop_size[0],
                                          w2h_ratio=0.75,
                                          p=0.5),
                                      dict(
                                          type='CenterCrop',
                                          height=crop_size[1],
                                          width=crop_size[0],
                                          p=0.5),
                                      dict(
                                          type='RandomResizedCrop',
                                          height=crop_size[1],
                                          width=crop_size[0],
                                          scale=(0.08, 1.0), 
                                          ratio=(0.75, 1.33),
                                          p=0.5),
                                  ],
                                  p=0.5
                              ),

                              dict(
                                  type='OneOf',
                                  transforms=[
                                      dict(
                                          type='RandomBrightnessContrast',
                                          brightness_limit=[0.05, 0.2],
                                          contrast_limit=[0.1, 0.3],
                                          p=0.5),
                                      dict(
                                          type='RandomGamma', gamma_limit=(80, 120), p=0.5),
                                      dict(
                                          type='CLAHE', clip_limit=0.4, tile_grid_size=(8, 8), p=0.5),
                                      dict(
                                          type='ToGray',  p=0.1),
                                  ],
                                  p=0.5
                              ),

                              dict(
                                  type='OneOf',
                                  transforms=[
                                      dict(
                                          type='RGBShift', r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),
                                      dict(type='ChannelShuffle', p=0.5),
                                  ],
                                  p=0.5
                              ),

                              dict(type='HorizontalFlip', p=0.5),
                              dict(type='VerticalFlip', p=0.1),
                              dict(type='Rotate', limit=15, interpolation=2,
                                   border_mode=0, value=0, p=0.5),


                              dict(
                                  type='OneOf',
                                  transforms=[
                                      dict(type='MotionBlur',
                                           blur_limit=7, p=0.2),
                                      dict(type='GaussianBlur',
                                           blur_limit=7, p=0.2),
                                      dict(type='MedianBlur',
                                           blur_limit=3, p=0.2),
                                      dict(type='Defocus', radius=(1, 5),
                                           alias_blur=(0.1, 0.3), p=0.2),
                                      dict(type='GlassBlur', sigma=0.3,
                                           max_delta=1, p=0.1),
                                      dict(type='Downscale', scale_min=0.5,
                                           scale_max=0.5, p=0.1),
                                      dict(type='ImageCompression', quality_lower=30,
                                           quality_upper=70, p=0.1),
                                  ],
                                  p=0.2),
                              dict(
                                  type='OneOf',
                                  transforms=[
                                      dict(type='ElasticTransform', alpha=20,
                                           sigma=3, alpha_affine=3, p=0.2),
                                      dict(type='GridDistortion', p=0.2),
                                  ],
                                  p=0.2
                              ),
                              dict(
                                  type='OneOf',
                                  transforms=[
                                      dict(
                                          type='CoarseDropout',
                                          max_holes=2,
                                          max_height=50,
                                          max_width=50,
                                          fill_value=0,
                                          p=0.2),
                                      dict(
                                          type='PixelDropout',
                                          dropout_prob=0.01,
                                          drop_value=0,
                                          p=0.2),
                                      dict(
                                          type='GridDropout',
                                          ratio=0.3,
                                          unit_size_min=5,
                                          unit_size_max=20,
                                          fill_value=0,
                                          p=0.2),
                                  ],
                                  p=0.2,
                              ),
                          ])

    test_pipeline = [
        # dict(type='RandomFlip', prob=0.5),
        # dict(type='RandomRotate', prob=0.4, degree=10, pad_val=0, seg_pad_val=255),
        albu_transform,
        # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='Resize', img_scale=crop_size, keep_ratio=False),
        # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    ]
    test_pipeline = Compose(test_pipeline)

    # crop_module = build_from_cfg(transform, PIPELINES)
    # results = crop_module(results)
    results = test_pipeline(results)
    gray_mask = results['gt_semantic_seg']*255
    # mmcv.imwrite(np.hstack((results['img'], cv2.merge(
    #     (gray_mask, gray_mask, gray_mask)))), "albu.jpg")
    print(results['img'].shape, " : ", gray_mask.shape)
    show_img = np.hstack(
        (results['img'], cv2.merge((gray_mask, gray_mask, gray_mask))))
    h, w = show_img.shape[0: 2]

    r = 0.7
    h = int(h * 0.7)
    w = int(w * 0.7)
    show_img = mmcv.imresize(show_img, (w, h))

    mmcv.imshow(show_img, "albu.jpg", wait_time=-1)


if __name__ == "__main__":
    print(__name__)
    test_albu()
