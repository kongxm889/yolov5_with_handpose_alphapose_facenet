"""Exports a pytorch *.pt model to *.onnx format

Usage:
    import torch
    $ export PYTHONPATH="$PWD" && python models/onnx_export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse

import onnx
# from alfred.dl.torch.common import device
from models.common import *
from utils.activations import Hardswish
from nb.torch.blocks.conv_blocks import ConvBase

device = torch.device("cuda")
print('Using device: ', device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[768, 1280], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    print(opt)

    # Parameters
    f = opt.weights.replace('.pt', '.onnx')  # onnx filename
    # chw
    print(opt.img_size)
    img = torch.randn((opt.batch_size, 3, opt.img_size[0], opt.img_size[1])).to(device) # image size, (1, 3, 320, 192) iDetection
    # img = img.half()

    # Load pytorch model
    assets = torch.load(opt.weights, map_location={'cpu': 'cuda'})
    # print(assets)
    # print('best fitness: ', assets['best_fitness'])
    model = assets['model']
    model.to(device)

    # Export to onnx
    try:
        model.model[-1].export = True  # set Detect() layer export=True
    except Exception as e:
        print('exception: {}, you may saved in DataParallel, using it normal.'.format(e))
        model = model.module
        model.model[-1].export = True

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
        if isinstance(m, ConvBase) and isinstance(m.activate, nn.Hardswish):
            # print('replacing Hardswish...')
            m.activate = Hardswish()  # assign activation
        # if isinstance(m, Detect):
        #    m.forward = m.forward_export  # assign forward (optional)    
        
    model.to(device)
    model.eval()
    # model.fuse()
    # model.half()
    
    # model.model[-1].m.to(device).half()
    o = model(img)  # dry run
    # print('output shape: {}'.format(o.shape))
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'], keep_initializers_as_inputs=False,
                      output_names=['output'], enable_onnx_checker=False)  # output_names=['output']

    # Check onnx model
    model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model)  # check onnx model
    print("The model after optimization:\n\n{}".format(
                onnx.helper.printable_graph(model.graph)))
    # print(onnx.helper.printable_graph(model.graph))  # print a human readable representation of the graph
    print('Export complete. ONNX model saved to %s\nView with https://github.com/lutzroeder/netron' % f)
