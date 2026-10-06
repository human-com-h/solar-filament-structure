import json, random, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset


def load_annotations(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    images = {im['id']: im for im in data['images']}
    anns_by_image = defaultdict(list)
    for ann in data['annotations']:
        anns_by_image[ann['image_id']].append(ann)
    return data, images, anns_by_image


def render_union_mask(image_id, images, anns_by_image, resolution=1024):
    im = images[image_id]
    h0, w0 = int(im['height']), int(im['width'])
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    sx, sy = resolution / w0, resolution / h0
    for ann in anns_by_image.get(image_id, []):
        for poly in ann.get('segmentation', []):
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if len(arr) < 3:
                continue
            pts = np.round(arr * np.array([sx, sy], dtype=np.float32)).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, resolution - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, resolution - 1)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def build_mask_cache(json_path, out_dir, resolution=1024):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data, images, anns_by_image = load_annotations(json_path)
    for i, image_id in enumerate(images):
        p = out_dir / f'{image_id}.png'
        if not p.exists():
            m = render_union_mask(image_id, images, anns_by_image, resolution)
            cv2.imwrite(str(p), m * 255)
        if (i + 1) % 100 == 0:
            print(f'cached {i+1}/{len(images)}')
    return len(images)


def deterministic_choice(items, seed, epoch, key):
    if len(items) == 1:
        return items[0]
    s = f'{seed}|{epoch}|{key}'.encode('utf-8')
    v = int(hashlib.sha256(s).hexdigest()[:16], 16)
    return items[v % len(items)]


class PhysicalObservationDataset(Dataset):
    """One physical observation = one training sample per epoch.

    For multi-annotator observations, one annotation copy is selected deterministically
    per epoch. This prevents duplicate observations from receiving 2-3x sampling weight.
    """
    def __init__(self, image_dir, json_path, manifest_csv, split, resolution=1024,
                 seed=20260831, mask_cache_dir=None):
        self.image_dir = Path(image_dir)
        self.resolution = int(resolution)
        self.seed = int(seed)
        self.epoch = 0
        self.data, self.images, self.anns_by_image = load_annotations(json_path)
        self.manifest = pd.read_csv(manifest_csv)
        self.manifest = self.manifest[self.manifest['split'] == split].copy()
        self.groups = []
        for physical_id, g in self.manifest.groupby('physical_observation_id', sort=True):
            records = g['annotation_sample_id'].tolist()
            file_name = g['file_name'].iloc[0]
            self.groups.append((str(physical_id), file_name, records))
        self.mask_cache_dir = Path(mask_cache_dir) if mask_cache_dir else None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.groups)

    def _mask(self, image_id):
        if self.mask_cache_dir is not None:
            p = self.mask_cache_dir / f'{image_id}.png'
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape != (self.resolution, self.resolution):
                        m = cv2.resize(m, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                    return (m > 127).astype(np.float32)
        return render_union_mask(image_id, self.images, self.anns_by_image, self.resolution).astype(np.float32)

    def __getitem__(self, idx):
        physical_id, file_name, records = self.groups[idx]
        image_id = deterministic_choice(records, self.seed, self.epoch, physical_id)
        with Image.open(self.image_dir / file_name) as img:
            img = img.convert('L').resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
            x = np.asarray(img, dtype=np.float32) / 255.0
        y = self._mask(image_id)
        x = torch.from_numpy(x[None, ...])
        y = torch.from_numpy(y[None, ...])
        return x, y, physical_id, image_id


class AnnotationCopyDataset(Dataset):
    """Validation/evaluation helper: one annotation copy per row."""
    def __init__(self, image_dir, json_path, manifest_csv, split, resolution=1024, mask_cache_dir=None):
        self.image_dir = Path(image_dir)
        self.resolution = int(resolution)
        self.data, self.images, self.anns_by_image = load_annotations(json_path)
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.mask_cache_dir = Path(mask_cache_dir) if mask_cache_dir else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        image_id = r.annotation_sample_id
        with Image.open(self.image_dir / r.file_name) as img:
            img = img.convert('L').resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
            x = np.asarray(img, dtype=np.float32) / 255.0
        if self.mask_cache_dir is not None and (self.mask_cache_dir / f'{image_id}.png').exists():
            m = cv2.imread(str(self.mask_cache_dir / f'{image_id}.png'), cv2.IMREAD_GRAYSCALE)
            if m.shape != (self.resolution, self.resolution):
                m = cv2.resize(m, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
            y = (m > 127).astype(np.float32)
        else:
            y = render_union_mask(image_id, self.images, self.anns_by_image, self.resolution).astype(np.float32)
        return torch.from_numpy(x[None, ...]), torch.from_numpy(y[None, ...]), str(r.physical_observation_id), image_id
