import argparse
import torch
from models.yolov6 import YoloV6
import yaml
from utils import torch_utils
from models.backbones.ghostnet import GhostNet
from utils.plots import color_list

import torch
# get list of models



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/custom/yolov5s_tl_v3.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    device = torch_utils.select_device(opt.device)

    # Create model
    # model = Model(opt.cfg).to(device)
    # model.train()

    model = YoloV6(opt.cfg).to(device)
    a = torch.randn([1, 3, 832, 832]).to(device)
    b = model(a)
    for a in b:
      print(a.shape)

    torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

    # load pretrained models, using ResNeSt-50 as an example
    net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

    a  =torch.randn([1, 3, 512, 512])
    net.eval()
    b = net(a)
    print(b)
