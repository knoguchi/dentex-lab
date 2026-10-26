"""DENTEX dataset loader for torchvision detection models.

DENTEX annotations are COCO-style JSON. Each box carries three labels:
  category_id_1 = quadrant (1-4), category_id_2 = tooth number (1-8),
  category_id_3 = diagnosis. We only use the diagnosis here.
"""
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

# index 0 is reserved for "background" by torchvision detection models
CLASSES = ["background", "Impacted", "Caries", "Periapical Lesion", "Deep Caries"]
NUM_CLASSES = len(CLASSES)


class DentexDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_dir, max_side=1333, train=False):
        self.image_dir = Path(image_dir)
        self.max_side = max_side
        self.train = train
        data = json.load(open(json_path))
        self.images = data["images"]
        # group annotations by image id
        self.anns = {}
        for a in data["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)
        # sanity check: category names match our CLASSES order
        names = [c["name"] for c in sorted(data["categories_3"], key=lambda c: c["id"])]
        assert names == CLASSES[1:], names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img = Image.open(self.image_dir / info["file_name"]).convert("RGB")
        w, h = img.size
        # shrink big x-rays up front so decoding/augment is cheaper
        scale = min(1.0, self.max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)

        boxes, labels = [], []
        for a in self.anns.get(info["id"], []):
            x, y, bw, bh = a["bbox"]  # COCO format: top-left + width/height
            boxes.append([x * scale, y * scale, (x + bw) * scale, (y + bh) * scale])
            labels.append(a["category_id_3"] + 1)  # +1 because 0 = background
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)

        img = to_tensor(img)  # float [0,1], shape (3, H, W)
        if self.train and random.random() < 0.5:  # horizontal flip augmentation
            img = img.flip(-1)
            W = img.shape[-1]
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]

        target = {"boxes": boxes, "labels": labels, "image_id": info["id"]}
        return img, target


def collate(batch):
    """Detection models take lists of images/targets (images can differ in size)."""
    return tuple(zip(*batch))
