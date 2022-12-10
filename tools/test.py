import argparse
import torchvision
import os
import os.path as osp
from mmdet.apis import init_detector, inference_detector

def save_fusion_result(config_file, checkpoint_file=None, img_dir=None):

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    out_filepath = '/root/DetFusion/demo/fusion_results'
    if not os.path.exists(out_filepath):
        os.mkdir(out_filepath)

    VIS_dir = osp.join(img_dir, 'VIS')
    TIR_dir = osp.join(img_dir, 'TIR')
    img_list = os.listdir(VIS_dir)

    for img_name in img_list:
        img_name_tir = img_name + '.jpg'
        img_name_rgb = img_name + '.jpg'
        img_name_fusion = img_name + '.jpg'
        img_path_i = osp.join(TIR_dir, img_name_tir)
        img_path_r = osp.join(VIS_dir, img_name_rgb)
        img_out_path = osp.join(out_filepath, img_name_fusion)
        image_fusion = inference_detector(model, img_path_r, img_path_i)
        torchvision.utils.save_image(image_fusion, img_out_path)
        print(img_out_path)

if __name__ == '__main__':
    save_fusion_result('/root/DetFusion/configs/DetFusion.py', '/root/DetFusion/work_dirs/DetFusion/model.pth', img_dir='/root/DetFusion/demo/')
