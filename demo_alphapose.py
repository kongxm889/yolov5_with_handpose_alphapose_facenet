import argparse
import platform
import time

import numpy as np
import torch
import cv2

from alphapose.models import builder
from alphapose.utils.config import update_config

from demo import detect_for_pose
from alfred.dl.torch.common import device


"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', default="weights/alphapose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml", type=str,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', default="weights/alphapose/fast_res50_256x192.pth", type=str,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=True)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=True)

args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(
    ',')] if torch.cuda.device_count() >= 1 else [-1]
print(args.gpus)
args.device = torch.device(
    "cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector == 'tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

input_size = (256, 192)
hm_out_size = (64, 48)


def get_inps(boxes, ori_img):
    inps = torch.zeros([len(boxes), 3, input_size[0], input_size[1]])
    bboxes = []
    scores = []
    for i, b in enumerate(boxes):
        # print(b)
        box = b[:4]
        x, y, xx, yy = box
        score = int(b[-2])
        if xx-x > 0 and yy-y > 0:
            o, box = test_transform(ori_img, box)
            inps[i] = o
            bboxes.append(box)
            scores.append(score)
    return inps, bboxes, scores


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_affine_transform(center, scale,
                         rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, trans):
    target_coords = np.zeros(coords.shape)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def get_max_pred_cuda_batched(heatmaps):
    v, i = torch.max(heatmaps, dim=2)
    maxvals, ii = torch.max(v, dim=2)
    iia = ii.unsqueeze(-1)
    iw = torch.gather(i, 2, iia)
    preds = torch.cat([iia, iw], dim=2)
    maxvals = maxvals.unsqueeze(-1)

    mask = maxvals > 0
    pred_mask = torch.cat([mask, mask], dim=2)
    preds *= pred_mask
    return preds, maxvals


def heatmap_to_coord_batched(hms_batched, bboxes):
    b, n_joints, hm_h, hm_w = hms_batched.shape
    batched_preds, batched_maxvals = get_max_pred_cuda_batched(hms_batched)
    # 20,17,2 20,17

    pose_coords = []
    pose_scores = []
    for bi in range(b):
        bbox = bboxes[bi]
        coords = batched_preds[bi]
        preds = np.zeros_like(coords)
        # transform bbox to scale
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        center = np.array([xmin + w * 0.5, ymin + h * 0.5])
        scale = np.array([w, h])
        trans = get_affine_transform(center, scale, 0, [hm_w, hm_h], inv=1)

        for i in range(coords.shape[0]):
            preds[i] = transform_preds(coords[i], trans)

        pose_coords.append(preds)
        s = batched_maxvals[bi]
        pose_scores.append(s)
    # print(pose_coords)
    # print(pose_scores)
    return pose_coords, pose_scores


def vis_frame(img, kps, kps_scores, format='coco'):
    kp_num = 17
    if kps[0].shape[0] > 0:
        kp_num = kps[0].shape[0]
    if kp_num == 17:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
        elif format == 'mpii':
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
        else:
            raise NotImplementedError
    elif kp_num == 136:
        raise NotImplementedError
    elif kp_num == 26:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),  # Foot
        ]
    else:
        raise NotImplementedError
    # c = (255, 158, 23)
    # c = (255, 0, 255)
    c = (0, 255, 0)
    # c = (255, 255, 0)
    for i in range(len(kps)):
        part_line = {}
        kp_preds = kps[i]
        kp_scores = kps_scores[i]
        if kp_num == 17:
            # kp_preds = torch.cat((kp_preds, torch.unsqueeze(
            #     (kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_preds = np.vstack([kp_preds, (kp_preds[5, :]+kp_preds[6, :])/2])
            # kp_scores = torch.cat((kp_scores, torch.unsqueeze(
            #     (kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
            kp_scores = np.vstack(
                [kp_scores, (kp_scores[5, :]+kp_scores[6, :])/2])

        # Draw keypoints
        vis_thres = 0.05 if kp_num == 136 else 0.4
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= vis_thres:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            cv2.circle(img, (int(cor_x), int(cor_y)),
                       1, c, 4, cv2.LINE_AA)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, c, 1, cv2.LINE_AA)
    return img


def vis_hm(ori_img, boxes, scores, hm):
    eval_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    if hm.size()[1] == 136:
        eval_joints = [*range(0, 136)]
    elif hm.size()[1] == 26:
        eval_joints = [*range(0, 26)]
    if len(boxes) == hm.shape[0]:
        t0 = time.time()
        pose_coords, pose_scores = heatmap_to_coord_batched(hm, boxes)
        t1 = time.time()
        print('vis_hm: ', t1-t0)
        ori_img = vis_frame(ori_img, pose_coords, pose_scores)
        return ori_img
    else:
        print('boxes and hm num not same, {} vs {}'.format(
            len(boxes), hm.shape[0]))


def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def _center_scale_to_box(center, scale):
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = torch.tensor(img).float()
    if img.max() > 1:
        img /= 255
    return img


def test_transform(src, bbox):
    xmin, ymin, xmax, ymax = bbox
    center, scale = _box_to_center_scale(
        xmin, ymin, xmax - xmin, ymax - ymin, float(input_size[1]) / input_size[0])
    # scale = scale * 1.0
    inp_h, inp_w = input_size
    trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
    img = cv2.warpAffine(
        src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
    bbox = _center_scale_to_box(center, scale)
    img = im_to_torch(img)
    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)
    return img, bbox


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    # init YOLOv5 detector
    weights = 'weights/yolov5s.pt'
    det_model = torch.load(weights, map_location=device)['model']
    det_model.to(device).eval()
    det_model.float()
    print('yolov5 model loaded from: ', weights)

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading pose model from {args.checkpoint}...')
    pose_model.load_state_dict(torch.load(
        args.checkpoint, map_location=args.device))
    pose_model.to(device)
    pose_model.eval()

    # Init data writer
    batchSize = args.posebatch

    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # (inps, orig_img, im_name, scores, ids, cropped_boxes) = det_loader.read()
        t0 = time.time()
        res = detect_for_pose(frame, det_model)
        if len(res) > 0:
            inps, bboxes, scores = get_inps(res, frame)
            t1 = time.time()
            print('detect in {}s'.format(t1 - t0))

            # Pose Estimation
            # inps = torch.cat(inps, dim=0)
            inps = inps.to(device)
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []

            for j in range(num_batches):
                inps_j = inps[j *
                              batchSize:min((j + 1) * batchSize, datalen)]
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            t2 = time.time()
            print('pose in {}s'.format(t2 - t1))

            hm = hm.cpu()
            res = vis_hm(frame, bboxes, scores, hm)
            t2 = time.time()
            res=cv2.resize(res,(1280,720))
            cv2.imshow('aa', res)
            cv2.waitKey(1)
