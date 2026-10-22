"""
Draw labels on a random sample of images so a human can check them by eye.

  python -m wase.inspect_labels                       # 50 random images, human labels
  python -m wase.inspect_labels --n 50 --seed 1       # a different random 50
  python -m wase.inspect_labels --teacher ..\\yolov8n.pt   # also draw teacher pseudo-labels
  python -m wase.inspect_labels --split dev --source rtts

Writes <out>/<sample-name>/:
  NN_<image>.jpg    the image with boxes drawn (human = solid, teacher = dashed)
  index.html        all images on one page with per-image details
  sample.json       which images were drawn and every box on them

What to look for:
  * boxes sit on the objects (not shifted, not scaled wrongly)
  * class names are right (bus vs motorcycle, bicycle vs person)
  * the three fog copies of one street carry the same boxes
  * 't=' is the box's fog transmittance; low t should look faint or invisible
"""

import argparse
import html
import random
from pathlib import Path

import cv2
import numpy as np

from .common import (CLASSES, class_map_for, default_data_dir, default_results_dir, imread,
                     load_json, log, nms_classwise, predict, save_json)

COLORS = {0: (60, 180, 255), 1: (255, 120, 40), 2: (80, 220, 80), 3: (220, 60, 220), 4: (40, 40, 240)}
MAX_WIDTH = 1600


def _dashed_rect(img, p1, p2, color, thick, dash=10):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thick)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thick)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thick)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thick)


def draw(img, human, teacher):
    H, W = img.shape[:2]
    thick = max(1, round(max(H, W) / 700))
    font = max(0.4, max(H, W) / 2600)
    for b in teacher or []:
        c, cx, cy, w, h, conf = b
        p1 = (int((cx - w / 2) * W), int((cy - h / 2) * H))
        p2 = (int((cx + w / 2) * W), int((cy + h / 2) * H))
        _dashed_rect(img, p1, p2, (255, 255, 255), thick)
        cv2.putText(img, f"T {CLASSES[c]} {conf:.2f}", (p1[0], p2[1] + int(14 * font * 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick, cv2.LINE_AA)
    for b in human:
        c, cx, cy, w, h, t = b
        col = COLORS[c]
        p1 = (int((cx - w / 2) * W), int((cy - h / 2) * H))
        p2 = (int((cx + w / 2) * W), int((cy + h / 2) * H))
        cv2.rectangle(img, p1, p2, col, thick)
        label = CLASSES[c] + ("" if t is None else f" t={t:.2f}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        y0 = max(th + 2, p1[1] - 2)
        cv2.rectangle(img, (p1[0], y0 - th - 2), (p1[0] + tw + 2, y0 + 2), col, -1)
        cv2.putText(img, label, (p1[0] + 1, y0), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=default_data_dir())
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="all", choices=["all", "train", "dev"])
    ap.add_argument("--source", default="all", choices=["all", "rtts", "city"])
    ap.add_argument("--teacher", default=None, help="also draw this model's pseudo-labels (dashed)")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=default_results_dir() / "inspect")
    a = ap.parse_args()

    data = a.data.resolve()
    manifest = load_json(data / "manifest.json")
    human = load_json(data / "human_boxes.json")
    pool = sorted(s for s, m in manifest.items()
                  if (a.split == "all" or m["split"] == a.split)
                  and (a.source == "all" or m["source"] == a.source))
    if not pool:
        raise SystemExit("no images match the filters")
    pick = random.Random(a.seed).sample(pool, min(a.n, len(pool)))   # random, not serial

    model = cmap = None
    if a.teacher:
        from ultralytics import YOLO
        model = YOLO(a.teacher)
        cmap = class_map_for(model)

    name = f"{a.split}_{a.source}_n{len(pick)}_seed{a.seed}" + ("_teacher" if a.teacher else "")
    out = (a.out / name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, record = [], []
    for k, s in enumerate(pick):
        m = manifest[s]
        img = imread(data / "store" / "images" / m["name"])
        if img is None:
            log(f"[warn] unreadable: {m['name']}")
            continue
        teach = nms_classwise(predict(model, img, a.conf, cmap)) if model else None
        img = draw(img, human[s], teach)
        if img.shape[1] > MAX_WIDTH:
            f = MAX_WIDTH / img.shape[1]
            img = cv2.resize(img, (MAX_WIDTH, int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
        fn = f"{k + 1:02d}_{Path(m['name']).stem}.jpg"
        cv2.imwrite(str(out / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        ts = [b[5] for b in human[s] if b[5] is not None]
        counts = {c: sum(1 for b in human[s] if b[0] == i) for i, c in enumerate(CLASSES)}
        desc = (f"{m['source']} · split {m['split']} · β={m['beta']} · {len(human[s])} human boxes"
                + (f" · min t {min(ts):.2f}" if ts else "")
                + (f" · {len(teach)} teacher boxes" if teach is not None else ""))
        rows.append((fn, m["name"], desc, ", ".join(f"{c} {n}" for c, n in counts.items() if n)))
        record.append({"image": m["name"], "stem": s, **{k2: m[k2] for k2 in ("source", "split", "beta", "scene")},
                       "human_boxes": human[s], "teacher_boxes": teach})

    legend = " ".join(f'<span style="background:rgb{tuple(reversed(c))};padding:1px 6px">{CLASSES[i]}</span>'
                      for i, c in COLORS.items())
    cards = "\n".join(
        f'<figure><a href="{html.escape(fn)}"><img src="{html.escape(fn)}" loading="lazy"></a>'
        f"<figcaption><b>{i + 1}. {html.escape(orig)}</b><br>{html.escape(d)}<br>{html.escape(cnt)}</figcaption></figure>"
        for i, (fn, orig, d, cnt) in enumerate(rows))
    page = f"""<!doctype html><meta charset="utf-8"><title>Label check</title>
<style>body{{font:14px system-ui;margin:16px;background:#f4f4f4}}figure{{background:#fff;margin:0 0 18px;padding:8px}}
img{{max-width:100%}}figcaption{{margin-top:4px}}</style>
<h2>Label check: {len(rows)} random images (seed {a.seed}, split {a.split}, source {a.source})</h2>
<p>Solid boxes = human labels, coloured by class: {legend}. {'Dashed white = teacher pseudo-labels (conf ≥ ' + str(a.conf) + ').' if a.teacher else ''}
<br>t = fog transmittance of the box (1 = fully visible, near 0 = hidden by fog; RTTS has no t).</p>
{cards}"""
    (out / "index.html").write_text(page, encoding="utf-8")
    save_json({"seed": a.seed, "split": a.split, "source": a.source, "images": record}, out / "sample.json")
    src = {"rtts": sum(r["source"] == "rtts" for r in record), "city": sum(r["source"] == "city" for r in record)}
    log(f"drew {len(rows)} images ({src['city']} Foggy Cityscapes, {src['rtts']} RTTS) -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
