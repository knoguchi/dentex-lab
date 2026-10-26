# dentex-lab — learn to train & run a dental disease detector (PyTorch)

[日本語版](README.ja.md)

Detects **Caries, Deep Caries, Periapical Lesion, Impacted tooth** on panoramic
dental x-rays. Object detection (bounding boxes), fine-tuned from a COCO-pretrained
Faster R-CNN (torchvision). Minimal learning project: three scripts, no app framework.

## Data

[DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) (MICCAI 2023 challenge,
CC BY-NC-SA 4.0). We use the fully-labelled subset only:

| split | images | boxes |
|-------|--------|-------|
| train | 705    | 3529 (Caries 2189, Impacted 604, Deep Caries 578, Periapical 158) |
| val   | 50     | 182   |

Full training zip is ~11 GB (mostly unlabeled). `scripts/download_data.py` reads the
zip over HTTP Range requests and extracts only the ~2.9 GB labelled subset.

```sh
uv sync
uv run python scripts/download_data.py            # train subset -> data/raw/training_data/
# validation set (small) — fetched separately:
uv run python -c "from huggingface_hub import hf_hub_download as d; [d('ibrahimhamamci/DENTEX', f'DENTEX/{f}', repo_type='dataset', local_dir='data/raw') for f in ['validation_data.zip','validation_triple.json']]"
(cd data/raw && unzip -q DENTEX/validation_data.zip -x '*/.ipynb_checkpoints/*')
```

## Train

```sh
uv run python train.py --epochs 10          # ~6 min/epoch on Apple M-series (MPS)
uv run python train.py --epochs 1 --limit 40   # smoke test
```

Saves best-val-mAP checkpoint to `checkpoints/fasterrcnn_dentex.pt`.

## Infer

```sh
uv run python infer.py data/raw/validation_data/quadrant_enumeration_disease/xrays/val_0.png --out result.png --thresh 0.5
```

Prints detections and writes an annotated image.

## Files

- `dental_data.py` — `Dataset` that turns DENTEX COCO-style JSON into `(image_tensor, {boxes, labels})`.
- `train.py` — model building, training loop, mAP evaluation (torchmetrics).
- `infer.py` — load checkpoint, run one image, draw boxes.
- `scripts/download_data.py` — remote-zip downloader that extracts only the needed files.

## Things worth understanding

- **Label 0 = background.** torchvision detection heads reserve class 0; we shift DENTEX ids by +1.
- **Boxes are `[x1, y1, x2, y2]`** in torchvision, but COCO JSON stores `[x, y, w, h]`. Convert.
- **Train mode returns losses, eval mode returns predictions.** Same `model(images, targets)` call.
- **Four losses**: RPN objectness + RPN box regression (proposals), classifier + box regression (final heads).
- **Resizing**: we shrink images to max side 1333 in the dataset; the model's internal `GeneralizedRCNNTransform` resizes again to min_size 800 anyway, and normalizes with ImageNet stats.
- **Augmentation**: only horizontal flip. Flipping a panoramic x-ray swaps left/right quadrants, which doesn't matter for diagnosis labels.
- **Metric**: mAP@[.5:.95] (COCO-style). Small/rare classes (Periapical Lesion, 158 boxes) will score lowest.

## Result of first run (10 epochs, MPS, ~50 min)

| epoch | train loss | val mAP@[.5:.95] | mAP@.5 | Impacted | Caries | Periapical | Deep Caries |
|------:|-----------:|-----------------:|-------:|---------:|-------:|-----------:|------------:|
| 0 | 0.699 | 0.172 | 0.276 | 0.45 | 0.18 | 0.01 | 0.05 |
| 3 | 0.359 | 0.280 | 0.448 | 0.52 | 0.19 | 0.17 | 0.25 |
| **5** | 0.291 | **0.316** | **0.481** | 0.55 | 0.25 | 0.21 | 0.26 |
| 9 | 0.166 | 0.293 | 0.436 | 0.55 | 0.25 | 0.14 | 0.23 |

Train loss keeps falling after epoch 5 while val mAP plateaus/drops: classic overfitting on 705 images.
Ideas to try next: stronger augmentation (brightness/contrast, scale jitter), lower LR, freeze more
backbone layers, or merge Caries + Deep Caries (the model confuses them constantly).
