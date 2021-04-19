import csv
import json
import os
import time

import cv2
import numpy as np

np.random.seed(0)


def PolyArea(x, y):
    """Calculate area of polygon given (x,y) coordinates (Shoelace formula)

    :param x: np.ndarray(N, )
    :param y: np.ndarray(N, )
    :return: area
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def save_coco_anno(data_annotation,
                   img_root,
                   save_path,
                   start_img_id=0,
                   start_ann_id=0,
                   kpt_num=17):
    """Save annotations in coco-format.

    :param data_annotation: list of data annotation.
    :param img_root: the root dir to load images.
    :param save_path: the path to save transformed annotation file.
    :param start_img_id: the starting point to count the image id.
    :param start_ann_id: the starting point to count the annotation id.
    :param kpt_num: the number of keypoint.
    """
    images = []
    annotations = []

    img_id = start_img_id
    ann_id = start_ann_id

    for i in range(0, len(data_annotation)):
        data_anno = data_annotation[i]
        image_name = data_anno[0]

        img = cv2.imread(os.path.join(img_root, image_name))

        kp_string = data_anno[1]
        kps = json.loads(kp_string)

        seg_string = data_anno[2]
        segs = json.loads(seg_string)

        for kp, seg in zip(kps, segs):
            keypoints = np.zeros([kpt_num, 3])
            for ind, p in enumerate(kp):
                if p['position'] is None:
                    continue
                else:
                    keypoints[ind, 0] = p['position'][0]
                    keypoints[ind, 1] = p['position'][1]
                    keypoints[ind, 2] = 2

            segmentation = np.array(seg[0]['segment'])
            max_x, max_y = segmentation.max(0)
            min_x, min_y = segmentation.min(0)

            anno = {}
            anno['keypoints'] = keypoints.reshape(-1).tolist()
            anno['image_id'] = img_id
            anno['id'] = ann_id
            anno['num_keypoints'] = int(sum(keypoints[:, 2] > 0))
            anno['bbox'] = [
                float(min_x),
                float(min_y),
                float(max_x - min_x + 1),
                float(max_y - min_y + 1)
            ]
            anno['iscrowd'] = 0
            anno['area'] = float(
                PolyArea(segmentation[:, 0], segmentation[:, 1]))
            anno['category_id'] = 1
            anno['segmentation'] = segmentation.reshape([1, -1]).tolist()

            annotations.append(anno)
            ann_id += 1

        image = {}
        image['id'] = img_id
        image['file_name'] = image_name
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]

        images.append(image)
        img_id += 1

    cocotype = {}

    cocotype['info'] = {}
    cocotype['info']['description'] = 'MacaquePose Generated by MMPose Team'
    cocotype['info']['version'] = '1.0'
    cocotype['info']['year'] = time.strftime('%Y', time.localtime())
    cocotype['info']['date_created'] = time.strftime('%Y/%m/%d',
                                                     time.localtime())

    cocotype['images'] = images
    cocotype['annotations'] = annotations
    cocotype['categories'] = [{
        'supercategory':
        'animal',
        'id':
        1,
        'name':
        'macaque',
        'keypoints': [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
            'right_knee', 'left_ankle', 'right_ankle'
        ],
        'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                     [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                     [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    }]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(cocotype, open(save_path, 'w'), indent=4)
    print('number of images:', img_id)
    print('number of annotations:', ann_id)
    print(f'done {save_path}')


dataset_dir = '/data/macaque/'

with open(os.path.join(dataset_dir, 'annotations.csv'), 'r') as fp:
    data_annotation_all = list(csv.reader(fp, delimiter=','))[1:]

np.random.shuffle(data_annotation_all)

data_annotation_train = data_annotation_all[0:12500]
data_annotation_val = data_annotation_all[12500:]

img_root = os.path.join(dataset_dir, 'images')
save_coco_anno(
    data_annotation_train,
    img_root,
    os.path.join(dataset_dir, 'annotations', 'macaque_train.json'),
    kpt_num=17)
save_coco_anno(
    data_annotation_val,
    img_root,
    os.path.join(dataset_dir, 'annotations', 'macaque_test.json'),
    start_img_id=12500,
    start_ann_id=15672,
    kpt_num=17)
