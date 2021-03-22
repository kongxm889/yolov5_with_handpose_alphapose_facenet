import os
import sys
import torch

from trt_quant.trt import TensorRTModel
from trt_quant.onnx_tensorrt import to_cuda

from torchvision.models import resnet18

import numpy as np
import os
import cv2

from alfred.utils.log import logger as logging
import glob

"""

this file will do the int8 quantization on onnx model
and generate int8 tensorrt engine.

"""

MAX_BATCH_SIZE = 1
CALIB_IMG_DIR = '/autox-sz/users/fagangjin/trafficlight_data/trafficlight_badcase_test/sh_test'



def letterbox3(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    ori_h, ori_w = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    INPUT_H, INPUT_W = new_shape
    r_w = INPUT_W / (ori_w * 1.0)
    r_h = INPUT_H / (ori_h * 1.0)
    if (r_h > r_w):
        w = INPUT_W
        h = r_w * ori_h
        x = 0
        y = (INPUT_H - h) / 2
    else:
        w = r_h * ori_w
        h = INPUT_H
        x = (INPUT_W - w) / 2
        y = 0
    
    img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(y - 0.1)), int(round(y + 0.1))
    left, right = int(round(x - 0.1)), int(round(x + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

class DataLoader:
    def __init__(self):
        self.index = 0
        self.length = 100
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(CALIB_IMG_DIR, "*.jpg"))
        assert len(self.img_list) > 100, '{} must contains more than 100 images to calib'.format(CALIB_IMG_DIR)
        logging.info('found all {} images to calib.'.format(len(self.img_list)))
        self.max_batch_size = 1

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.length:
            self.index += 1
            data = []
            for i in range(self.max_batch_size):
                assert os.path.exists(self.img_list[self.index]), 'not found!!'
                img = cv2.imread(self.img_list[self.index])
                data.append(img)
            # example only
            return data, data
        else:
            raise StopIteration

    def __len__(self):
        return self.length


def preprocess(inputs):
    def letterbox2(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = new_shape
            ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        # import pdb
        # pdb.set_trace()

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        # cv2.imshow('rr', img_new)
        # cv2.waitKey(0)
        return img_new
    # img list
    data = inputs[0]
    # must return dict
    # preprocess for every single img inside list
    new_d = []
    for im in data:
        img = letterbox2(im, new_shape=[768, 1280])
        img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)
        # print('img: ---- ', img)
        new_d.append(img)
    data = torch.tensor(new_d)
    # print('data input shape: {}'.format(data.shape))
    return {
        "images": to_cuda(data)/255.,
    }


if __name__ == '__main__':
    # onnx_f = 'weights/v5s_no_csp/v5s_no_csp_sim.onnx'
    # onnx_f = 'weights/v5s_no_csp/best_sim.onnx'
    if len(sys.argv) <= 1:
        print('provide an onnx file.')
        exit(0)
    onnx_f = sys.argv[1]
    if 'sim' not in onnx_f:
        print('Are you sure this onnx is simplified? onnx must simplified before convert to trt engine.')
        exit(0)
    engine_f = os.path.join(os.path.dirname(onnx_f), os.path.basename(onnx_f).split('.')[0] + '.trt')
    cache_f = os.path.join(os.path.dirname(onnx_f), "int8_calib.txt")
    
    max_calibration_batch = 512
    data_loader = DataLoader()
    int8_calibrator = TensorRTModel.get_int8_calibrator(
        max_calibration_batch, data_loader, preprocess, cache_f)
    TensorRTModel.build_engine(onnx_f, engine_f, 4, device="CUDA",
                               fp16_mode=False, int8_mode=True, int8_calibrator=int8_calibrator)
