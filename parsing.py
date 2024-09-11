import torch
import numpy as np
import cv2
from PIL import Image 
from torch.utils import data 
import os 
from .transforms import get_affine_transform, transform_logits


def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def delete_irregular(logits_result):
    parsing_result = np.argmax(logits_result, axis=2)
    upper_cloth = np.where(parsing_result == 4, 255, 0)
    contours, hierarchy = cv2.findContours(upper_cloth.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i], True)
        area.append(abs(a))
    if len(area) != 0:
        top = area.index(max(area))
        M = cv2.moments(contours[top])
        cY = int(M["m01"] / M["m00"])

    dresses = np.where(parsing_result == 7, 255, 0)
    contours_dress, hierarchy_dress = cv2.findContours(dresses.astype(np.uint8),
                                                       cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area_dress = []
    for j in range(len(contours_dress)):
        a_d = cv2.contourArea(contours_dress[j], True)
        area_dress.append(abs(a_d))
    if len(area_dress) != 0:
        top_dress = area_dress.index(max(area_dress))
        M_dress = cv2.moments(contours_dress[top_dress])
        cY_dress = int(M_dress["m01"] / M_dress["m00"])
    wear_type = "dresses"
    if len(area) != 0:
        if len(area_dress) != 0 and cY_dress > cY:
            irregular_list = np.array([4, 5, 6])
            logits_result[:, :, irregular_list] = -1
        else:
            irregular_list = np.array([5, 6, 7, 8, 9, 10, 12, 13])
            logits_result[:cY, :, irregular_list] = -1
            wear_type = "cloth_pant"
        parsing_result = np.argmax(logits_result, axis=2)
    # pad border
    parsing_result = np.pad(parsing_result, pad_width=1, mode='constant', constant_values=0)
    return parsing_result, wear_type



def hole_fill(img):
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)
        # keep large area in skin case
        for j in range(len(area)):
          if j != i and area[i] > 2000:
             cv2.drawContours(refine_mask, contours, j, color=255, thickness=-1)
    return refine_mask

def refine_hole(parsing_result_filled, parsing_result, arm_mask):
    filled_hole = cv2.bitwise_and(np.where(parsing_result_filled == 4, 255, 0),
                                  np.where(parsing_result != 4, 255, 0)) - arm_mask * 255
    contours, hierarchy = cv2.findContours(filled_hole, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    refine_hole_mask = np.zeros_like(parsing_result).astype(np.uint8)
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i], True)
        # keep hole > 2000 pixels
        if abs(a) > 2000:
            cv2.drawContours(refine_hole_mask, contours, i, color=255, thickness=-1)
    return refine_hole_mask + arm_mask

class SimpleFolderDataset(data.Dataset):
    def __init__(self, root, input_size=[512, 512], transform=None):
        self.root = root
        self.input_size = input_size
        self.transform = transform
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.input_size = np.asarray(input_size)
        self.is_pil_image = False
        if isinstance(root, Image.Image):
            self.file_list = [root]
            self.is_pil_image = True
        elif os.path.isfile(root):
            self.file_list = [os.path.basename(root)]
            self.root = os.path.dirname(root)
        else:
            self.file_list = os.listdir(self.root)

    def __len__(self):
        return len(self.file_list)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

def __getitem__(self, index):
    if self.is_pil_image:
        img = np.asarray(self.file_list[index])[:, :, [2, 1, 0]]
    else:
        img_name = self.file_list[index]
        img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    # Get person center and scale
    person_center, s = self._box2cs([0, 0, w - 1, h - 1])
    r = 0
    trans = get_affine_transform(person_center, s, r, self.input_size)
    input = cv2.warpAffine(
        img,
        trans,
        (int(self.input_size[1]), int(self.input_size[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))

    input = self.transform(input)
    meta = {
        'center': person_center,
        'height': h,
        'width': w,
        'scale': s,
        'rotation': r
    }

    return input, meta

