from demo import detect_for_pose
from facenet import *

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    # init YOLOv5 detector
    weights = 'weights/yolov5s.pt'
    det_model = torch.load(weights, map_location=device)['model']
    det_model.to(device).eval()
    det_model.float()
    print('yolov5 model loaded from: ', weights)

    # Load face model
    face_model = Facenet()

    cap = cv2.VideoCapture("D:/PythonProjects/flask_yolov5_and_facenet/video3.mp4")
    video_index = 0
    while (cap.isOpened()):
        fpstime = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        # (inps, orig_img, im_name, scores, ids, cropped_boxes) = det_loader.read()
        t0 = time.time()
        res = detect_for_pose(frame, det_model, draw=False)
        if not len(res):
            continue
        facenet_input = []
        aligned_test = []
        person_img_boxes = []
        person_img_bestprobs = []

        # 对每个有人的框检测人脸
        for index, (box_left, box_top, box_right, box_bottom, conf) in enumerate(res):
            if box_bottom - box_top < min_person_size:
                continue

            face_prebox = frame[int(box_top):int(
                min(box_bottom, box_top + (box_right - box_left) / face_input_size_w * face_input_size_h)),
                          int(box_left):int(box_right)]

            padding_h_need = int(
                face_prebox.shape[1] / face_input_size_w * face_input_size_h - face_prebox.shape[0])
            if padding_h_need > 0:
                face_prebox = np.vstack((face_prebox, np.zeros(
                    (padding_h_need, face_prebox.shape[1], 3), np.uint8)))
            face_prebox = cv2.resize(face_prebox, (face_input_size_w, face_input_size_h))

            facenet_input.append(Image.fromarray(face_prebox))
            person_img_boxes.append([box_left, box_top, box_right, box_bottom])
        probs, boxes, faces = face_model.face_infer(facenet_input)  # 把人的上面一部分放进去识别人脸

        # 给每个人画框
        for person_img_box in person_img_boxes:
            c1, c2 = (int(person_img_box[0]), int(person_img_box[1])), (
                int(person_img_box[2]), int(person_img_box[3]))

            cv2.rectangle(frame, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        # print("=========faces=========\n", faces)
        # print("=========boxes=========\n",boxes)
        # print("=========probs=========\n",probs)
        # 对有人脸的行人框，把人脸信息加入列表
        faceindex2nameindex_list = []  # 可查询第i张face是第j张图中的
        for index, pre_person_img_prob in enumerate(probs):
            if boxes[index] is not None:  # 这个人框里面有人脸
                pre_person_bestface_index = np.argmax(pre_person_img_prob)  # 找置信度最大的框
                aligned_test.append(faces[index][pre_person_bestface_index])
                faceindex2nameindex_list.append(index)
                person_img_bestprobs.append(max(pre_person_img_prob))

        # 对每张图的所有人脸，检测是谁
        if len(aligned_test):
            aligned_test = torch.stack(aligned_test).to(device)
            embeddings_test = face_model.resnet(aligned_test).detach().cpu()
            dists = [[(e1 - e2).norm().item() for e2 in face_model.embeddings] for e1 in embeddings_test]
            # print("=========dists=========\n",dists)
            names = [
                "N" if min(dist) > face2name_config else face_model.dataset.idx_to_class[dist.index(min(dist))]
                for
                dist in dists]
            # 把差不多的人脸加入库
            # for index, name in enumerate(names):
            #     if name != "N":
            #         facenet.embeddings.append(embeddings_test[index])
            #         facenet.dataset.idx_to_class[len(facenet.embeddings)] = name

            # print("=========names=========\n"," ".join(names))
            # 画框 写名字
            for index, name in enumerate(names):
                person_img_box = person_img_boxes[faceindex2nameindex_list[index]]
                c1, c2 = (int(person_img_box[0]), int(person_img_box[1])), (
                    int(person_img_box[2]), int(person_img_box[3]))

                # cv2.rectangle(yolo_result, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                if person_img_bestprobs[index] > draw_yolo_config:
                    if name != '?':
                        tf = max(tl - 1, 1)  # font thickness
                        t_size = cv2.getTextSize(name, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.rectangle(frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(frame, name, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
                                    thickness=tf,
                                    lineType=cv2.LINE_AA)
        cv2.putText(frame, str(video_index), (10, 60), 0, tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        # 写入mp4前缩小
        frame = cv2.resize(frame, save_video_size)
        cv2.imshow("img", frame)
        if cv2.waitKey(1) == 27:
            break
        print("\ryolo_face_fps:{}".format(int(1 / (time.time() - fpstime))))
