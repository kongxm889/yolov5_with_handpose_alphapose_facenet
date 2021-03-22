import os
from argparse import ArgumentParser

from xtcocotools.coco import COCO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from demo import detect_for_pose
from alfred.dl.torch.common import device
import cv2
import time
import torch


def process_mmdet_results(mmdet_results, cat_id=0):
    person_results = []
    for bbox in mmdet_results:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)
    return person_results


def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--video', type=str, default='', help='Image root')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()


    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())
    weights = 'yolov5s.pt'
    det_model = torch.load(weights, map_location=device)['model']
    det_model.to(device).eval()
    det_model.float()
    print('yolov5 model loaded from: ', weights)

    dataset = pose_model.cfg.data['test']['type']
    return_heatmap = False
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    cap = cv2.VideoCapture(args.video)

    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        tic = time.time()
        res = detect_for_pose(img, det_model)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(res)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=0.3,
            format='xyxy',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        cost = time.time() - tic
        print('cost: {}, fps: {}'.format(cost, 1/cost))
        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=False)

        cv2.imshow('Image', vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
