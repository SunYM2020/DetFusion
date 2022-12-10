import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation
from .rotate_aug import RotateAugmentation
from .rotate_aug import RotateTestAugmentation

import cv2

class LLVIP_TSCustomDataset(Dataset):
    """Two Stream Custom dataset for detection.
    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 rotate_aug=None,
                 rotate_test_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.img_infos, self.rgb_img_infos= self.load_annotations(ann_file)

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None

        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            self.rgb_img_infos = [self.rgb_img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # if use rotation augmentation
        if rotate_aug is not None:
            self.rotate_aug = RotateAugmentation(self.CLASSES, **rotate_aug)
        else:
            self.rotate_aug = None

        if rotate_test_aug is not None:
            #  dot not support argument settings currently
            self.rotate_test_aug = RotateTestAugmentation()
        else:
            self.rotate_test_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_info_r = self.rgb_img_infos[idx]
        img_info_i = self.img_infos[idx]

        img_r = mmcv.imread(osp.join(self.img_prefix, 'visible/train', img_info_r['filename']))
        img_i = mmcv.imread(osp.join(self.img_prefix, 'infrared/train', img_info_i['filename']))

        img_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        newimg = np.zeros_like(img_r)
        newimg[:,:,0] = img_gray
        newimg[:,:,1] = img_gray
        newimg[:,:,2] = img_gray
        img_r = newimg

        source_img_r = img_gray
        source_img_r = mmcv.imresize(source_img_r, (640, 512), return_scale=False)
        source_img_r = source_img_r[:,:,np.newaxis].astype(np.float32)

        source_img_i = img_i[:,:,0]
        source_img_i = mmcv.imresize(source_img_i, (640, 512), return_scale=False)
        source_img_i = source_img_i[:,:,np.newaxis].astype(np.float32)
        
        source_img_r = source_img_r.transpose(2, 0, 1)
        source_img_i = source_img_i.transpose(2, 0, 1)

        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]

            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']

        if self.with_mask:
            gt_masks = ann['masks']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']
        
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img_r, gt_bboxes, gt_labels = self.extra_aug(img_r, gt_bboxes,
                                                    gt_labels)
            img_i, gt_bboxes, gt_labels = self.extra_aug(img_i, gt_bboxes,
                                                    gt_labels)

        # rotate augmentation
        if self.rotate_aug is not None:
            img_r, gt_bboxes, gt_masks, gt_labels = self.rotate_aug(img_r, gt_bboxes,
                                                                    gt_masks, gt_labels, img_info_r['filename'])
            img_i, gt_bboxes, gt_masks, gt_labels = self.rotate_aug(img_i, gt_bboxes,
                                                                    gt_masks, gt_labels, img_info_i['filename'])

            gt_bboxes = np.array(gt_bboxes).astype(np.float32)

            if len(gt_bboxes) == 0:
                return None

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale 
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        # RGB
        img_r, img_shape_r, pad_shape_r, scale_factor_r = self.img_transform(
            img_r, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img_r = img_r.copy()
        # TIR
        img_i, img_shape_i, pad_shape_i, scale_factor_i = self.img_transform(
            img_i, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img_i = img_i.copy()

        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info_r['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]

        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape_i, scale_factor_i,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals

        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape_r, scale_factor_r,
                                        flip)

        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape_r,
                                                scale_factor_r, flip)

        if self.with_mask:
            gt_masks = self.mask_transform(gt_masks, pad_shape_r,
                                        scale_factor_r, flip)

        ori_shape = (img_info_i['height'], img_info_i['width'], 3)
        img_meta_r = dict(
            ori_shape=ori_shape,
            img_shape=img_shape_r,
            pad_shape=pad_shape_r,
            scale_factor=scale_factor_r,
            flip=flip)

        data = dict(
            img_r=DC(to_tensor(img_r), stack=True),
            img_i=DC(to_tensor(img_i), stack=True),
            img_meta=DC(img_meta_r, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)),
            source_img_r=DC(to_tensor(source_img_r), stack=True), 
            source_img_i=DC(to_tensor(source_img_i), stack=True)) 

        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info_r = self.rgb_img_infos[idx]
        img_info_i = self.img_infos[idx]

        img_r = mmcv.imread(osp.join('/root/DetFusion/demo/fusion_results', img_info_r['filename']))
        img_i = mmcv.imread(osp.join('/root/DetFusion/demo/fusion_results', img_info_i['filename']))

        source_img_r = img_r[:,:,0]
        source_img_r = source_img_r[:,:,np.newaxis].astype(np.float32)

        source_img_i = img_i[:,:,0]
        source_img_i = source_img_i[:,:,np.newaxis].astype(np.float32)
        
        source_img_r = source_img_r.transpose(2, 0, 1)
        source_img_i = source_img_i.transpose(2, 0, 1)

        img_r = mmcv.imresize(img_r, (1280, 1024), return_scale=False)
        img_i = mmcv.imresize(img_i, (1280, 1024), return_scale=False)

        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img_r, img_i, scale, flip, proposal=None):

            _img_r, img_shape_r, pad_shape_r, scale_factor_r = self.img_transform(
                img_r, scale, flip, keep_ratio=self.resize_keep_ratio)

            _img_i, img_shape_i, pad_shape_i, scale_factor_i = self.img_transform(
                img_i, scale, flip, keep_ratio=self.resize_keep_ratio)

            _img_r = to_tensor(_img_r)
            _img_i = to_tensor(_img_i)

            _img_meta_r = dict(
                ori_shape=(img_info_r['height'], img_info_r['width'], 3),
                img_shape=img_shape_r,
                pad_shape=pad_shape_r,
                scale_factor=scale_factor_r,
                flip=flip,
                angle=0)
            _img_meta_i = dict(
                ori_shape=(img_info_i['height'], img_info_i['width'], 3),
                img_shape=img_shape_i,
                pad_shape=pad_shape_i,
                scale_factor=scale_factor_i,
                flip=flip,
                angle=0)

            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img_r, _img_i, _img_meta_r, _img_meta_i, _proposal

        def prepare_rotation_single(img_r, img_i, scale, flip, angle):

            _img_r, img_shape_r, pad_shape_r, scale_factor_r = self.rotate_test_aug(
                img_r, angle=angle)
            _img_r, img_shape_r, pad_shape_r, scale_factor_r = self.img_transform(
                _img_r, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img_r = to_tensor(_img_r)

            _img_i, img_shape_i, pad_shape_i, scale_factor_i = self.rotate_test_aug(
                img_i, angle=angle)
            _img_i, img_shape_i, pad_shape_i, scale_factor_i = self.img_transform(
                _img_i, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img_i = to_tensor(_img_i)

            _img_meta_r = dict(
                ori_shape=(img_info_r['height'], img_info_r['width'], 3),
                img_shape=img_shape_r,
                pad_shape=pad_shape_r,
                scale_factor=scale_factor_r,
                flip=flip,
                angle=angle
            )
            _img_meta_i = dict(
                ori_shape=(img_info_i['height'], img_info_i['width'], 3),
                img_shape=img_shape_i,
                pad_shape=pad_shape_i,
                scale_factor=scale_factor_i,
                flip=flip,
                angle=angle
            )
            return _img_r, _img_i, _img_meta_r, _img_meta_i

        imgs_r = []
        img_metas_r = []
        imgs_i = []
        img_metas_i = []
        proposals = []

        for scale in self.img_scales:
            _img_r, _img_i, _img_meta_r, _img_meta_i, _proposal = prepare_single(
                img_r, img_i, scale, False, proposal)

            imgs_r.append(_img_r)
            img_metas_r.append(DC(_img_meta_r, cpu_only=True))
            imgs_i.append(_img_i)
            img_metas_i.append(DC(_img_meta_i, cpu_only=True))
            proposals.append(_proposal)

            if self.flip_ratio > 0:
                _img_r, _img_i, _img_meta_r, _img_meta_i, _proposal = prepare_single(
                    img_r, img_i, scale, True, proposal)
                imgs_r.append(_img_r)
                img_metas_r.append(DC(_img_meta_r, cpu_only=True))
                imgs_i.append(_img_i)
                img_metas_i.append(DC(_img_meta_i, cpu_only=True))
                proposals.append(_proposal)

        if self.rotate_test_aug is not None :
            for angle in [90, 180, 270]:

                for scale in self.img_scales:
                    _img_r, _img_i, _img_meta_r, _img_meta_i = prepare_rotation_single(
                        img_r, img_i, scale, False, angle)
                    imgs_r.append(_img_r)
                    img_metas_r.append(DC(_img_meta_r, cpu_only=True))
                    imgs_i.append(_img_i)
                    img_metas_i.append(DC(_img_meta_i, cpu_only=True))

                    if self.flip_ratio > 0:
                        _img_r, _img_i, _img_meta_r, _img_meta_i = prepare_rotation_single(
                            img_r, img_i, scale, True, proposal, angle)
                        imgs_r.append(_img_r)
                        img_metas_r.append(DC(_img_meta_r, cpu_only=True))
                        imgs_i.append(_img_i)
                        img_metas_i.append(DC(_img_meta_i, cpu_only=True))

        data = dict(img_r=imgs_r, img_i=imgs_i, img_meta=img_metas_r, source_img_r=source_img_r, source_img_i=source_img_i)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
