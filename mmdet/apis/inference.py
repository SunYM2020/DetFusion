import warnings

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.models import build_detector
import cv2

def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def _inference_generator(model, imgs, img_transform, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, device)


def inference_detector(model, imgs_r, imgs_i):
    cfg = model.cfg
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device
    if not isinstance(imgs_r, list):
        return _inference_single(model, imgs_r, imgs_i, img_transform, device)
    else:
        return _inference_generator(model, imgs, img_transform, device)


def _prepare_data(img_r, img_i, img_transform, cfg, device, imgid):
    ori_shape_r = img_r.shape
    ori_shape_i = img_i.shape

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

    img_r, img_shape_r, pad_shape_r, scale_factor_r = img_transform(
        img_r,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img_i, img_shape_i, pad_shape_i, scale_factor_i = img_transform(
        img_i,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img_r = to_tensor(img_r).to(device).unsqueeze(0)
    img_i = to_tensor(img_i).to(device).unsqueeze(0)
    source_img_r = to_tensor(source_img_r).to(device).unsqueeze(0)
    source_img_i = to_tensor(source_img_i).to(device).unsqueeze(0)
    
    img_meta_r = [
        dict(
            ori_shape=ori_shape_r,
            img_shape=img_shape_r,
            pad_shape=pad_shape_r,
            scale_factor=scale_factor_r,
            img_id=imgid,
            flip=False)
    ]

    return dict(img_r=[img_r], img_i=[img_i], img_meta=[img_meta_r], source_img_r=[source_img_r], source_img_i=[source_img_i])


def _inference_single(model, img_r, img_i, img_transform, device):
    imgid = '1'
    img_r = mmcv.imread(img_r)
    img_i = mmcv.imread(img_i)
    data = _prepare_data(img_r, img_i, img_transform, model.cfg, device, imgid)
    with torch.no_grad():
        result_r, result_i, image_fusion = model(return_loss=False, rescale=True, **data)
    return image_fusion


def show_result(img, result, class_names, score_thr=0.3, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file)

def draw_poly_detections(img, detections, class_names, scale, threshold=0.2, putText=False,showStart=False, colormap=None):
    """

    :param img:
    :param detections:
    :param class_names:
    :param scale:
    :param cfg:
    :param threshold:
    :return:
    """
    import pdb
    import cv2
    import random
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    color_white = (255, 255, 255)

    for j, name in enumerate(class_names):
        if colormap is None:
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        else:
            color = colormap[j]
        try:
            dets = detections[j]
        except:
            pdb.set_trace()
        for det in dets:
            bbox = det[:8] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = list(map(int, bbox))
            if showStart:
                cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
            for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=color, thickness=2,lineType=cv2.LINE_AA)
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2,lineType=cv2.LINE_AA)
            if putText:
                cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                            color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return img

