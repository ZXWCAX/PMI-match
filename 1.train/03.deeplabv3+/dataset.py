import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def cv2_imread_unicode(filepath, flags=cv2.IMREAD_COLOR):
    """Reads image files containing Unicode (e.g., Chinese) characters in paths."""
    if not os.path.exists(filepath):
        return None
    return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), flags)


class CADVectorFieldDataset(Dataset):
    def __init__(self, json_path, base_dir, target_size=1024):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.records = json.load(f)
        self.base_dir = base_dir
        self.target_size = target_size

        # ImageNet normalization statistics
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        folder_path = os.path.join(self.base_dir, record['id'])

        # ==================================================
        # 1. Load Raw Data
        # ==================================================
        input_npz_path = os.path.join(folder_path, "input.npz")
        input_data = np.load(input_npz_path)['tensor']
        rgb = input_data[:, :, :3]  # uint8
        bbox_mask = input_data[:, :, 3]  # uint8

        target_mask_path = os.path.join(folder_path, "target_mask.png")
        target_mask = cv2_imread_unicode(target_mask_path, cv2.IMREAD_GRAYSCALE)

        # ==================================================
        # 2. Proportional Scaling and Padding (Letterbox)
        # ==================================================
        orig_H, orig_W = rgb.shape[:2]
        scale = self.target_size / max(orig_H, orig_W)
        new_H, new_W = int(orig_H * scale), int(orig_W * scale)

        # Use INTER_AREA for RGB to preserve fine lines; INTER_NEAREST for masks to maintain binarization
        rgb_resized = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_AREA)
        bbox_mask_resized = cv2.resize(bbox_mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        target_mask_resized = cv2.resize(target_mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

        # Calculate padding dimensions (black borders)
        pad_h = self.target_size - new_H
        pad_w = self.target_size - new_W

        rgb_padded = np.pad(rgb_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        bbox_mask_padded = np.pad(bbox_mask_resized, ((0, pad_h), (0, pad_w)), mode='constant')
        target_mask_padded = np.pad(target_mask_resized, ((0, pad_h), (0, pad_w)), mode='constant')

        # ==================================================
        # 3. Compute Distance Field at 1024 Scale
        # ==================================================
        if cv2.countNonZero(target_mask_padded) == 0:
            dist_map = np.full((self.target_size, self.target_size), self.target_size, dtype=np.float32)
        else:
            inv_mask = cv2.bitwise_not(target_mask_padded)
            dist_map = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)

        # ==================================================
        # 4. Format Conversion and Normalization
        # ==================================================
        rgb_norm = rgb_padded.astype(np.float32) / 255.0
        rgb_norm = np.transpose(rgb_norm, (2, 0, 1))
        rgb_norm = (rgb_norm - self.mean) / self.std

        bbox_mask_norm = bbox_mask_padded.astype(np.float32) / 255.0
        bbox_mask_tensor = np.expand_dims(bbox_mask_norm, axis=0)

        input_tensor = np.concatenate([rgb_norm, bbox_mask_tensor], axis=0)
        dist_tensor = np.expand_dims(dist_map, axis=0)

        return {
            "input": torch.from_numpy(input_tensor).float(),
            "distance_map": torch.from_numpy(dist_tensor).float(),
            "bbox_mask": torch.from_numpy(bbox_mask_tensor).float()
        }






















