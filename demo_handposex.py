import argparse
import os

import cv2
import torch
import torch.nn.functional as F
from alfred.dl.torch.common import device
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_5

from demo import detect_for_pose

try:
    from handposex.models.resnet import resnet50, resnet34
    from handposex.models.squeezenet import squeezenet1_1, squeezenet1_0
    from handposex.models.rexnet import ReXNetV1
    from handposex.utils.common_utils import *
    from handposex.hand_data_iter.datasets import draw_bd_handpose
except ImportError as e:
    print('You need install handposex from manaai.cn, you can softlink under yolov5 is you have it already, ERROR: ', e)

torch.set_grad_enabled(False)

input_size = (256, 256)


def prepare_for_handclass_inp(boxes, img):
    h, w, c = img.shape
    inps = torch.zeros([len(boxes), 3, 192, 192])
    boxes = np.array(boxes).astype(np.int)
    for i, b in enumerate(boxes):
        # print(b)
        box = b[:4]
        box = refine_box(box, w, h)
        x, y, xx, yy = box
        score = int(b[-2])
        if xx - x > 0 and yy - y > 0:
            o = img[y: yy, x: xx]
            # print(o.shape)
            # cv2.imshow('aa', o)
            # cv2.waitKey(0)
            img_ = cv2.resize(
                o, (192, 192), interpolation=cv2.INTER_CUBIC)
            img_ = img_.astype(np.float32)
            img_ = (img_ - 128.) / 256.

            inps[i] = torch.tensor(img_).permute(2, 0, 1)
    return inps


def prepare_inp(boxes, img):
    h, w, c = img.shape
    inps = torch.zeros([len(boxes), 3, input_size[0], input_size[1]])
    bboxes = []
    scores = []
    boxes = np.array(boxes).astype(np.int)
    for i, b in enumerate(boxes):
        # print(b)
        box = b[:4]
        box = refine_box(box, w, h)
        x, y, xx, yy = box
        score = int(b[-2])
        if xx - x > 0 and yy - y > 0:
            o = img[y: yy, x: xx]
            # print(o.shape)
            # cv2.imshow('aa', o)
            # cv2.waitKey(0)
            img_ = cv2.resize(
                o, (input_size[1], input_size[0]), interpolation=cv2.INTER_CUBIC)
            img_ = img_.astype(np.float32)
            img_ = (img_ - 128.) / 256.

            inps[i] = torch.tensor(img_).permute(2, 0, 1)
            bboxes.append(box)
            scores.append(score)
    return inps, bboxes, scores


def refine_box(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    w_ = max(abs(x_max - x_min), abs(y_max - y_min))
    w_ = w_ * 1.2

    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2

    x1, y1, x2, y2 = int(x_mid - w_ / 2), int(y_mid - w_ /
                                              2), int(x_mid + w_ / 2), int(y_mid + w_ / 2)

    x1 = np.clip(x1, 0, img_width - 1)
    x2 = np.clip(x2, 0, img_width - 1)

    y1 = np.clip(y1, 0, img_height - 1)
    y2 = np.clip(y2, 0, img_height - 1)
    return [x1, y1, x2, y2]


def pred_pose(model, batched_inps, boxes):
    # batched_inps = batched_inps.transpose(2, 0, 1)
    batched_inps = batched_inps.to(device)
    print('hand pose input: ', batched_inps.shape)
    pre_ = model(batched_inps)
    output = pre_.cpu().detach().numpy()
    # output = np.squeeze(output)

    res = []
    idx = 0
    if len(output.shape) == 1:
        output = np.expand_dims(output, 0)
    print(output.shape)
    for oo in output:
        x1, y1, x2, y2 = boxes[idx]
        img_width = x2 - x1
        img_height = y2 - y1
        idx += 1

        pts_hand = {}
        if len(oo.shape) != 0:
            for i in range(int(oo.shape[0] / 2)):
                x = (oo[i * 2 + 0] * float(img_width))
                y = (oo[i * 2 + 1] * float(img_height))

                pts_hand[str(i)] = {}
                pts_hand[str(i)] = {
                    "x": x + x1,
                    "y": y + y1,
                }
            res.append(pts_hand)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' Project Hand Pose Inference')

    parser.add_argument('--model_path', type=str, default='handposex/weight_handpose/shufflenetv2_x1_5.pth',
                        help='model_path')  # 模型路径
    parser.add_argument('--model', type=str, default='shufflenetv2_x1_5',
                        help='model : resnet_x,squeezenet_x')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=42,
                        help='num_classes')  # 手部21关键点， (x,y)*2 = 42
    parser.add_argument('--GPUS', type=str, default='0',
                        help='GPUS')  # GPU选择
    parser.add_argument('--test_path', type=str, default='./image/',
                        help='test_path')  # 测试图片路径
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')  # 输入模型图片尺寸

    ops = parser.parse_args()  # 解析添加参数
    unparsed = vars(ops)  # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key, unparsed[key]))

    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    test_path = ops.test_path  # 测试图片文件夹路径
    print('use model : %s' % (ops.model))

    if ops.model == 'resnet_50':
        model_ = resnet50(num_classes=ops.num_classes,
                          img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_ = resnet34(num_classes=ops.num_classes,
                          img_size=ops.img_size[0])
    elif ops.model == "squeezenet1_0":
        model_ = squeezenet1_0(num_classes=ops.num_classes)
    elif ops.model == "squeezenet1_1":
        model_ = squeezenet1_1(num_classes=ops.num_classes)
    elif ops.model == 'shufflenetv2_x1_0':
        model_ = shufflenet_v2_x1_0(
            pretrained=False, num_classes=ops.num_classes)
    elif ops.model == 'shufflenetv2_x1_5':
        model_ = shufflenet_v2_x1_5(
            pretrained=False, num_classes=ops.num_classes)
    elif ops.model == 'rexnetv1':
        model_ = ReXNetV1(classes=ops.num_classes)
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval()

    if os.access(ops.model_path, os.F_OK):  # checkpoint
        chkpt = torch.load(ops.model_path, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.model_path))

    weights = 'weights/hand_v5s/best.pt'
    det_model = torch.load(weights, map_location=device)['model']
    det_model.to(device).eval()
    det_model.float()

    hand_class_model = resnet34(num_classes=14, img_size=192).to(device).eval()

    chkpt = torch.load("weights/resnet_34-size-192.pth", map_location=device)
    hand_class_model.load_state_dict(chkpt)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    while cap.isOpened():
        ret, itm = cap.read()
        if not ret:
            break
        # cv2.imshow('raw', itm)
        res = detect_for_pose(itm, det_model)
        if len(res) > 0:
            inps, boxes, scores = prepare_inp(res, itm)

            hand_class_inps = prepare_for_handclass_inp(res, itm)

            # 预测手骨架
            hand_pts = pred_pose(model_, inps, boxes)

            # 人手分类
            pre_hand_class = hand_class_model(hand_class_inps.cuda().float())

            outputs = F.softmax(pre_hand_class, dim=1)
            outputs = outputs.cpu().detach().numpy()

            hand_class_list = ["one", "five", "fist", "ok", "heartSingle", "yearh", "three", "four", "six", "Iloveyou",
                               "gun", "thumbUp", "nine", "pink"]
            hand_class_result = []
            for output in outputs:
                max_index = np.argmax(output)
                score_ = output[max_index]
                print('pre {}\tconfidence {}'.format(hand_class_list[max_index], score_))
                if score_ > 0.5:
                    hand_class_result.append(hand_class_list[max_index])
            hand_class_result_str = " ".join(hand_class_result)
            tl = 10
            tf = 10
            cv2.putText(itm, hand_class_result_str, (10, 100), 0, tl / 3, [255, 000, 255],
                        thickness=tf,
                        lineType=cv2.LINE_AA)

            for pts in hand_pts:
                draw_bd_handpose(itm, pts, 0, 0)
                for k, v in pts.items():
                    x = v['x']
                    y = v['y']
                    cv2.circle(itm, (int(x), int(y)), 2,
                               (255, 0, 255), -1, cv2.LINE_AA)
                    # cv2.circle(itm, (int(x), int(y)), 1, (255, 150, 180), -1)

        cv2.imshow('res', itm)
        cv2.waitKey(1)
