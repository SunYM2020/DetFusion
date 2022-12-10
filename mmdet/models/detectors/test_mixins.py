import torch
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_proposals,
                        merge_aug_bboxes, merge_aug_masks, multiclass_nms,
                        choose_best_Rroi_batch)
from mmdet.core import merge_rotate_aug_bboxes, merge_rotate_aug_proposals

class RPNTestMixin(object):

    def simple_test_rpn_r(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head_r(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head_r.get_bboxes(*proposal_inputs)
        return proposal_list

    def simple_test_rpn_i(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head_i(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head_i.get_bboxes(*proposal_inputs)
        return proposal_list

class BBoxTestMixin(object):

    def simple_test_bboxes_r(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor_r(
            x[:len(self.bbox_roi_extractor_r.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head_r(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head_r.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def simple_test_bboxes_i(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor_i(
            x[:len(self.bbox_roi_extractor_i.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head_i(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head_i.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels    
