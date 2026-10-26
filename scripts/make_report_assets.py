"""Build assets for docs/index.html: GT-vs-prediction images, per-class stats,
confusion matrix and the training curve parsed from checkpoints/train.log.

Usage: uv run python scripts/make_report_assets.py
"""
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dental_data import CLASSES, DentexDataset, collate  # noqa: E402
from train import VAL_IMG, VAL_JSON, build_model, get_device  # noqa: E402

OUT = Path("docs/assets")
THRESH = 0.5
IOU = 0.5
WEB_W = 1400
COLORS = {"Impacted": "#c98500", "Caries": "#e66767", "Periapical Lesion": "#199e70", "Deep Caries": "#9085e9"}
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)


def parse_log(path):
    pat = re.compile(r"== epoch (\d+): avg loss ([\d.]+)\s+val mAP@\[.5:.95\]=([\d.]+)\s+mAP@.5=([\d.]+)\s+(\{.*?\})")
    rows = []
    for line in open(path):
        m = pat.search(line)
        if m:
            rows.append({"epoch": int(m[1]), "loss": float(m[2]), "map": float(m[3]),
                         "map50": float(m[4]), "per_class": eval(m[5])})
    return rows


def draw(img, boxes, labels, scores=None, dashed=False):
    """Draw boxes on a PIL image already scaled to web size."""
    d = ImageDraw.Draw(img)
    for i, (box, lab) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        col = COLORS[lab]
        if dashed:  # ground truth: dashed outline
            step = 14
            for x in range(int(x1), int(x2), step):
                d.line([(x, y1), (min(x + 7, x2), y1)], fill=col, width=3)
                d.line([(x, y2), (min(x + 7, x2), y2)], fill=col, width=3)
            for y in range(int(y1), int(y2), step):
                d.line([(x1, y), (x1, min(y + 7, y2))], fill=col, width=3)
                d.line([(x2, y), (x2, min(y + 7, y2))], fill=col, width=3)
        else:
            d.rectangle([x1, y1, x2, y2], outline=col, width=3)
        text = lab if scores is None else f"{lab} {scores[i]:.2f}"
        tw = d.textlength(text, font=FONT)
        ty = y1 - 26 if y1 > 30 else y2 + 2
        d.rectangle([x1, ty, x1 + tw + 10, ty + 26], fill=col)
        d.text((x1 + 5, ty + 1), text, fill="black", font=FONT)
    return img


def main():
    device = get_device()
    ckpt = torch.load("checkpoints/fasterrcnn_dentex.pt", map_location="cpu")
    model = build_model(len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ds = DentexDataset(VAL_JSON, VAL_IMG)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate, num_workers=2)

    n = len(CLASSES)
    conf = [[0] * n for _ in range(n)]  # rows = GT class, cols = predicted class (0 = missed / spurious)
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0} for c in CLASSES[1:]}
    cases = []  # per image record for the showcase
    with torch.no_grad():
        for images, targets in loader:
            preds = model([im.to(device) for im in images])
            for img, t, p in zip(images, targets, preds):
                keep = p["scores"] > THRESH
                pb, pl, ps = p["boxes"][keep].cpu(), p["labels"][keep].cpu(), p["scores"][keep].cpu()
                gb, gl = t["boxes"], t["labels"]
                for c in gl.tolist():
                    per_class[CLASSES[c]]["gt"] += 1
                # greedy match: highest IoU first, each GT and pred used once
                matched_g, matched_p = set(), set()
                if len(gb) and len(pb):
                    ious = box_iou(gb, pb)
                    pairs = sorted(((ious[i, j].item(), i, j) for i in range(len(gb)) for j in range(len(pb))), reverse=True)
                    for iou, i, j in pairs:
                        if iou < IOU or i in matched_g or j in matched_p:
                            continue
                        matched_g.add(i); matched_p.add(j)
                        conf[gl[i]][pl[j]] += 1
                        if gl[i] == pl[j]:
                            per_class[CLASSES[gl[i]]]["tp"] += 1
                        else:
                            per_class[CLASSES[gl[i]]]["fn"] += 1
                            per_class[CLASSES[pl[j]]]["fp"] += 1
                for i in range(len(gb)):
                    if i not in matched_g:
                        conf[gl[i]][0] += 1; per_class[CLASSES[gl[i]]]["fn"] += 1
                for j in range(len(pb)):
                    if j not in matched_p:
                        conf[0][pl[j]] += 1; per_class[CLASSES[pl[j]]]["fp"] += 1

                fname = next(im["file_name"] for im in ds.images if im["id"] == t["image_id"])
                cases.append({"file": fname, "img": img, "gb": gb, "gl": gl, "pb": pb, "pl": pl, "ps": ps,
                              "n_gt": len(gb), "n_classes": len(set(gl.tolist()))})

    # showcase: prefer images with several boxes and >=2 classes; take 6
    cases.sort(key=lambda c: (c["n_classes"], c["n_gt"]), reverse=True)
    show = cases[:6]
    showcase = []
    for c in show:
        img = c["img"]
        _, h, w = img.shape
        scale = WEB_W / w
        base = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype("uint8")).resize((WEB_W, round(h * scale)))
        stem = Path(c["file"]).stem
        gt = draw(base.copy(), (c["gb"] * scale).tolist(), [CLASSES[i] for i in c["gl"].tolist()], dashed=True)
        pr = draw(base.copy(), (c["pb"] * scale).tolist(), [CLASSES[i] for i in c["pl"].tolist()], c["ps"].tolist())
        both = draw(gt.copy(), (c["pb"] * scale).tolist(), [CLASSES[i] for i in c["pl"].tolist()], c["ps"].tolist())
        base.save(OUT / f"{stem}_raw.jpg", quality=82)
        gt.save(OUT / f"{stem}_gt.jpg", quality=82)
        pr.save(OUT / f"{stem}_pred.jpg", quality=82)
        both.save(OUT / f"{stem}_both.jpg", quality=82)
        showcase.append({
            "stem": stem, "w": WEB_W, "h": round(h * scale),
            "gt": [{"cls": CLASSES[l]} for l in c["gl"].tolist()],
            "pred": [{"cls": CLASSES[l], "score": round(s, 2)} for l, s in zip(c["pl"].tolist(), c["ps"].tolist())],
        })

    for c, s in per_class.items():
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        r = s["tp"] / s["gt"] if s["gt"] else 0.0
        s["precision"], s["recall"] = round(p, 3), round(r, 3)

    report = {
        "classes": CLASSES, "colors": COLORS, "thresh": THRESH, "iou": IOU,
        "epochs": parse_log("checkpoints/train.log"),
        "best_epoch": ckpt["epoch"], "best_map": round(ckpt["map"], 3),
        "per_class": per_class, "confusion": conf, "showcase": showcase,
        "n_val_images": len(ds),
    }
    (Path("docs") / "data.js").write_text("window.REPORT = " + json.dumps(report, indent=1) + ";\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("showcase", "epochs")}, indent=1))
    print("showcase:", [s["stem"] for s in showcase])


if __name__ == "__main__":
    main()
