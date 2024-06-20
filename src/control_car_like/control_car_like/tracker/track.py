import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import requests, torch, math, cv2
import numpy as np
import PIL
from numpy import random
#Change directory so that imports wortk correctly
# if os.getcwd()=="/content":
#   os.chdir("YOLOv6")
from ..yolov6.utils.events import LOGGER, load_yaml
from ..yolov6.layers.common import DetectBackend
from ..yolov6.data.data_augment import letterbox
from ..yolov6.utils.nms import non_max_suppression
from ..yolov6.core.inferer import Inferer

from ..utils import visualization as vis
from .sort import *
from typing import List, Optional

device:str = "cpu"
half:bool = False #@param {type:"boolean"}
img_size:int = 640#@param {type:"integer"}
conf_thres: float = .25  # @param {type:"number"}
iou_thres: float = .45  # @param {type:"number"}
max_det: int = 1000  # @param {type:"integer"}
agnostic_nms: bool = False  # @param {type:"boolean"}
unique_track_color: bool = False
thickness: int = 2

# img_size = check_img_size(img_size, s=stride)
sort_tracker = Sort(max_age=5,
                    min_hits=2,
                    iou_threshold=0.2)

model = DetectBackend(f"src/control_car_like/control_car_like/data/humman.pt", device=device)
stride = model.stride
class_names = load_yaml("src/control_car_like/control_car_like/data/humman.yaml")['names']
colors = [[random.randint(0, 255) for _ in range(3)] for i in range(10000)]

def precess_image(img_src, img_size, stride, half):

    image = letterbox(img_src, img_size, stride=stride)[0]

    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src

def draw_boxes(img, bbox, identities=None, categories=None, confidences = None, names=None, colors=None):
    object_ = {}
    area_max = 0
    for i, box in enumerate(bbox):
        # color = (randint(0, 255), randint(0, 255), randint(0, 255))
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        color = colors[id]
    # if not opt.nobbox:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        # if not opt.nolabel:
        label = str(id) + ":"+ names[cat] if identities is not None else  f'{names[cat]} {confidences[i]:.2f}'
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        # center = ((x2-x1)//2, (y2-y1)//2)
        area = (x2-x1)*(y2-y1)
        object_[id] = area
    return img, object_


def tracking_sort(url):
    box_obj = []
    area = {}
    img, img_src = precess_image(url, img_size, stride, half)
    img = img.to(device)
    if len(img.shape) == 3:
        img = img[None]
        # expand for batch dim
    pred_results = model(img)
    classes: Optional[List[int]] = None  # the classes to keep
    det = non_max_suppression(pred_results, conf_thres,
                            iou_thres, classes, agnostic_nms, max_det=max_det)[0]
    # print(det)
    gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    img_ori = img_src.copy()
    if len(det):
        det[:, :4] = Inferer.rescale(
            img.shape[2:], det[:, :4], img_src.shape).round()

        dets_to_sort = np.empty((0, 6))
        for *xyxy, conf, cls in reversed(det):
            if int(cls)==0:
            # print(class_num)
                dets_to_sort = np.vstack((dets_to_sort,
                                    np.array([*xyxy, conf, cls])))

            # print(dets_to_sort)
        tracked_dets = sort_tracker.update(dets_to_sort, unique_track_color)
        tracks = sort_tracker.getTrackers()

        if len(tracked_dets) > 0:
            bbox_xyxy = tracked_dets[:, :4]
            identities = tracked_dets[:, 8]
            categories = tracked_dets[:, 4]
            confidences = None
            
            for t, track in enumerate(tracks):
                
                track_color = colors[int(
                    track.detclass)] if not unique_track_color else sort_tracker.color_list[t]

                [cv2.line(img_src, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])),
                        (int(track.centroidarr[i+1][0]),
                        int(track.centroidarr[i+1][1])),
                        track_color, thickness=thickness)
                for i, _ in enumerate(track.centroidarr)
                if i < len(track.centroidarr)-1]
                box_obj.append(track.centroidarr)

            img_ori, area = draw_boxes(img_src, bbox_xyxy, identities, categories, confidences, class_names, colors)
    # if bbox_xyxy is None:
    
    return box_obj, img_ori, area