"""Fine-tune a COCO-pretrained Faster R-CNN on DENTEX dental x-rays.

Usage: uv run python train.py --epochs 10
"""
import argparse
import time
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dental_data import CLASSES, NUM_CLASSES, DentexDataset, collate

TRAIN_JSON = "data/raw/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
TRAIN_IMG = "data/raw/training_data/quadrant-enumeration-disease/xrays"
VAL_JSON = "data/raw/DENTEX/validation_triple.json"
VAL_IMG = "data/raw/validation_data/quadrant_enumeration_disease/xrays"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes):
    # Start from COCO weights (backbone + RPN + box head already know "objectness").
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    # Swap the final classifier so it predicts our classes instead of COCO's 91.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    for images, targets in loader:
        images = [im.to(device) for im in images]
        preds = model(images)
        preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
        gts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(preds, gts)
    res = metric.compute()
    per_class = {}
    if "map_per_class" in res and res["map_per_class"].ndim > 0:
        for cls_idx, ap in zip(res["classes"].tolist(), res["map_per_class"].tolist()):
            per_class[CLASSES[cls_idx]] = round(ap, 3)
    return float(res["map"]), float(res["map_50"]), per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="use only N train images (smoke test)")
    ap.add_argument("--out", default="checkpoints/fasterrcnn_dentex.pt")
    args = ap.parse_args()

    device = get_device()
    print("device:", device)

    train_ds = DentexDataset(TRAIN_JSON, TRAIN_IMG, train=True)
    val_ds = DentexDataset(VAL_JSON, VAL_IMG)
    if args.limit:
        train_ds = torch.utils.data.Subset(train_ds, range(args.limit))
    print(f"train images: {len(train_ds)}  val images: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=2, shuffle=False, num_workers=args.workers, collate_fn=collate)

    model = build_model(NUM_CLASSES).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # warm up for 1 epoch then cosine decay — standard detection recipe
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=1 / max(args.epochs, 2), anneal_strategy="cos")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_map = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, (images, targets) in enumerate(train_loader):
            images = [im.to(device) for im in images]
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
            # In train mode torchvision detection models return a dict of losses
            # (rpn objectness, rpn box reg, classifier, box reg).
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 20 == 0:
                parts = " ".join(f"{k.replace('loss_', '')}={v.item():.3f}" for k, v in loss_dict.items())
                print(f"ep {epoch} step {step}/{steps_per_epoch} loss={loss.item():.3f} [{parts}] "
                      f"lr={scheduler.get_last_lr()[0]:.4f}", flush=True)
        train_time = time.time() - t0

        mAP, mAP50, per_class = evaluate(model, val_loader, device)
        print(f"== epoch {epoch}: avg loss {running / steps_per_epoch:.3f}  "
              f"val mAP@[.5:.95]={mAP:.3f}  mAP@.5={mAP50:.3f}  {per_class}  "
              f"({train_time:.0f}s train)", flush=True)
        if mAP > best_map:
            best_map = mAP
            torch.save({"model": model.state_dict(), "classes": CLASSES, "epoch": epoch, "map": mAP}, args.out)
            print(f"   saved {args.out}")
    print("best val mAP:", round(best_map, 3))


if __name__ == "__main__":
    main()
