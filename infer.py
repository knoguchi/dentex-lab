"""Run the trained detector on one x-ray and draw the findings.

Usage: uv run python infer.py path/to/xray.png [--out result.png] [--thresh 0.5]
"""
import argparse

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes

from train import build_model, get_device

FONT = next((f for f in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
             if __import__("os").path.exists(f)), None)
COLORS = {"Impacted": "yellow", "Caries": "red", "Periapical Lesion": "cyan", "Deep Caries": "magenta"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ckpt", default="checkpoints/fasterrcnn_dentex.pt")
    ap.add_argument("--out", default="result.png")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    img = Image.open(args.image).convert("RGB")
    x = to_tensor(img).to(device)
    with torch.no_grad():
        pred = model([x])[0]  # dict with boxes (x1,y1,x2,y2), labels, scores

    keep = pred["scores"] > args.thresh
    boxes = pred["boxes"][keep].cpu()
    labels = [classes[i] for i in pred["labels"][keep].tolist()]
    scores = pred["scores"][keep].tolist()

    for lab, sc, box in zip(labels, scores, boxes.tolist()):
        print(f"{lab:18s} {sc:.2f}  box={[round(v) for v in box]}")
    if not labels:
        print(f"no findings above threshold {args.thresh}")

    canvas = (to_tensor(img) * 255).to(torch.uint8)
    canvas = draw_bounding_boxes(
        canvas, boxes, labels=[f"{l} {s:.2f}" for l, s in zip(labels, scores)],
        colors=[COLORS[l] for l in labels], width=4, font=FONT, font_size=28)
    to_pil_image(canvas).save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
