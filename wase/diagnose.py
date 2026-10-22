"""
Mechanism studies that need no training (plan Block A). Each writes one JSON
under results/diagnose/.

  # A1-A3: scale dose-response, with and without DCP, unconditional confidence
  python -m wase.diagnose scale --weights yolov8n.pt --gt <gt.json> --img-root <dir>

  # A4: does the DQS choice pick the better layout, image by image?
  python -m wase.diagnose selection --weights yolov8n.pt --gt <gt.json> --img-root <dir>

  # A5: calibration (ECE, reliability bins) of any models
  python -m wase.diagnose calibration --models coco=yolov8n.pt naive=runs/naive/seed0/weights/final.pt \
      --gt <gt.json> --img-root <dir>

  # A6: teacher pseudo-labels vs human labels on the corpus, by size/visibility/source
  python -m wase.diagnose agreement --weights yolov8n.pt --split train --conf 0.15

Run scale/selection with two teachers (COCO and the naive seed-0 final model),
as the plan asks, by calling them twice with different --weights and --tag.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .common import (NC, class_map_for, default_data_dir, default_results_dir, dehaze, dqs, imread,
                     iou_xyxy, load_json, log, nms_classwise, predict, save_json)
from .evaluate import load_gt, resolve_images
from .labeling import refine_crop, refine_inplace
from .metrics import Scorer, expected_calibration_error, greedy_match

CANVAS = 640
GREY = 114


def _gt_xyxy(g):
    return [(int(c), b[0], b[1], b[0] + b[2], b[1] + b[3])
            for c, b, cr in zip(g["cls"], g["boxes"], g["crowd"]) if not cr]


def _best_gt(box_xyxy, cls, gts, thr=0.1):
    best, out = thr, None
    for g in gts:
        if g[0] == cls:
            v = iou_xyxy(box_xyxy, g[1:5])
            if v > best:
                best, out = v, g
    return out


def _to_xyxy_px(b, W, H):
    return ((b[1] - b[3] / 2) * W, (b[2] - b[4] / 2) * H, (b[1] + b[3] / 2) * W, (b[2] + b[4] / 2) * H)


def _summ(rows):
    if not rows:
        return {"n": 0}
    cand = [r for r in rows if r["conf_new"] is not None]
    acc = [r for r in cand if r["conf_new"] > r["conf_raw"]]
    gtc = [r for r in cand if r["iou_raw"] is not None and r["iou_new"] is not None]
    gta = [r for r in acc if r["iou_raw"] is not None and r["iou_new"] is not None]

    def m(xs):
        return float(np.mean(xs)) if xs else None

    def med(xs):
        return float(np.median(xs)) if xs else None
    return {
        "n_uncertain": len(rows),
        "redetected_rate": len(cand) / len(rows),
        "accept_rate_conf_rule": len(acc) / len(rows),
        "dconf_unconditional_mean": m([r["conf_new"] - r["conf_raw"] for r in cand]),
        "dconf_accepted_mean": m([r["conf_new"] - r["conf_raw"] for r in acc]),
        "area_ratio_median_all": med([r["area_ratio"] for r in cand]),
        "area_ratio_median_accepted": med([r["area_ratio"] for r in acc]),
        "iou_gt_raw_mean_all": m([r["iou_raw"] for r in gtc]),
        "iou_gt_new_mean_all": m([r["iou_new"] for r in gtc]),
        "iou_gt_raw_mean_accepted": m([r["iou_raw"] for r in gta]),
        "iou_gt_new_mean_accepted": m([r["iou_new"] for r in gta]),
        "pct_accepted_with_worse_iou": (100 * float(np.mean([r["iou_new"] < r["iou_raw"] for r in gta]))
                                        if gta else None),
        "n_with_gt": len(gtc), "n_accepted_with_gt": len(gta),
    }


# ---------------------------------------------------------------------------
def study_scale(a):
    from ultralytics import YOLO
    model = YOLO(a.weights)
    cmap = class_map_for(model)
    gt, files, _ = load_gt(a.gt)
    imgs = resolve_images(files, a.img_root)
    lo, hi = (float(x) for x in a.band.split(","))
    scales = [s if s == "letterbox" else float(s) for s in a.scales.split(",")]
    rows = defaultdict(list)
    skipped_big = defaultdict(int)
    for iid, path in tqdm(imgs.items(), desc="scale study"):
        img = imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = _gt_xyxy(gt[iid])
        raw = nms_classwise(predict(model, img, a.conf, cmap))
        unc = [b for b in raw if lo <= b[5] <= hi]
        for b in unc:
            rx = _to_xyxy_px(b, W, H)
            g = _best_gt(rx, b[0], gts)
            iou_raw = iou_xyxy(rx, g[1:5]) if g else None
            x1, y1 = max(0, int(rx[0])), max(0, int(rx[1]))
            x2, y2 = min(W, int(rx[2])), min(H, int(rx[3]))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            for use_dcp in (True, False):
                crop = img[y1:y2, x1:x2]
                crop = dehaze(crop) if use_dcp else crop
                for s in scales:
                    key = f"crop_s{s if s == 'letterbox' else f'{s:g}'}_{'dcp' if use_dcp else 'raw'}"
                    if s == "letterbox":
                        cands = predict(model, crop, 0.01, cmap)
                        mapped = [(c[0], x1 + (c[1] - c[3] / 2) * (x2 - x1), y1 + (c[2] - c[4] / 2) * (y2 - y1),
                                   x1 + (c[1] + c[3] / 2) * (x2 - x1), y1 + (c[2] + c[4] / 2) * (y2 - y1), c[5])
                                  for c in cands]
                    else:
                        cw, ch = int(round((x2 - x1) * s)), int(round((y2 - y1) * s))
                        if cw > CANVAS or ch > CANVAS:
                            skipped_big[key] += 1
                            continue
                        canvas = np.full((CANVAS, CANVAS, 3), GREY, np.uint8)
                        ox, oy = (CANVAS - cw) // 2, (CANVAS - ch) // 2
                        canvas[oy:oy + ch, ox:ox + cw] = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_LINEAR)
                        cands = predict(model, canvas, 0.01, cmap, imgsz=CANVAS)
                        mapped = [(c[0], x1 + ((c[1] - c[3] / 2) * CANVAS - ox) / s,
                                   y1 + ((c[2] - c[4] / 2) * CANVAS - oy) / s,
                                   x1 + ((c[1] + c[3] / 2) * CANVAS - ox) / s,
                                   y1 + ((c[2] + c[4] / 2) * CANVAS - oy) / s, c[5]) for c in cands]
                    same = sorted((m for m in mapped if m[0] == b[0]), key=lambda m: -m[5])
                    best = same[0] if same else None
                    rows[key].append({
                        "conf_raw": b[5], "conf_new": best[5] if best else None,
                        "area_ratio": ((best[3] - best[1]) * (best[4] - best[2]) /
                                       max((rx[2] - rx[0]) * (rx[3] - rx[1]), 1e-9)) if best else None,
                        "iou_raw": iou_raw, "iou_new": iou_xyxy(best[1:5], g[1:5]) if (best and g) else None})
        for use_dcp in (True, False):
            key = f"inplace_{'dcp' if use_dcp else 'raw'}"
            _, recs = refine_inplace(model, img, raw, cmap, (lo, hi), True, 640, use_dcp=use_dcp)
            for b, r in zip([x for x in raw if lo <= x[5] <= hi], recs):
                rx = _to_xyxy_px(b, W, H)
                g = _best_gt(rx, b[0], gts)
                bb = _to_xyxy_px(r["best_box"], W, H) if r["best_box"] else None
                rows[key].append({"conf_raw": r["conf_raw"], "conf_new": r["conf_best"],
                                  "area_ratio": r["area_ratio_best"],
                                  "iou_raw": iou_xyxy(rx, g[1:5]) if g else None,
                                  "iou_new": iou_xyxy(bb, g[1:5]) if (bb and g) else None})
    out = {"weights": a.weights, "gt": str(a.gt), "band": [lo, hi], "teacher_conf": a.conf,
           "note": ("s=k pastes the crop upscaled k-fold on a grey 640 canvas (constant input size); "
                    "s=1 removes context without changing scale; 'letterbox' is the previous round's "
                    "crop behaviour. conf_new is the best same-class re-detection whether or not the "
                    "keep-if-more-confident rule would accept it (A3). In-place candidates must overlap "
                    "the raw box (IoU >= 0.5) to count as a re-detection."),
           "skipped_exceeds_canvas": dict(skipped_big),
           "results": {k: _summ(v) for k, v in sorted(rows.items())}}
    save_json(out, a.out / f"scale_{a.tag}.json")
    log(f"wrote {a.out / f'scale_{a.tag}.json'}")


# ---------------------------------------------------------------------------
def study_selection(a):
    from ultralytics import YOLO
    model = YOLO(a.weights)
    cmap = class_map_for(model)
    gt, files, _ = load_gt(a.gt)
    imgs = resolve_images(files, a.img_root)
    band = tuple(float(x) for x in a.band.split(","))
    refiners = {"crop": refine_crop, "inplace": refine_inplace}
    layouts = {k: {} for k in ("raw", "crop_refined", "inplace_refined", "crop_dqs", "inplace_dqs")}
    sizes, per_image = {}, []
    for iid, path in tqdm(imgs.items(), desc="selection study"):
        img = imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        sizes[iid] = (W, H)
        g = _gt_xyxy(gt[iid])
        raw = nms_classwise(predict(model, img, a.conf, cmap))
        layouts["raw"][iid] = raw
        rec = {"image_id": iid}
        for kind, refine in refiners.items():
            refined = nms_classwise(refine(model, img, raw, cmap, band, True, 640)[0]) if raw else []
            chose = dqs(refined) > dqs(raw)          # the previous round's DQS duel
            layouts[f"{kind}_refined"][iid] = refined
            layouts[f"{kind}_dqs"][iid] = refined if chose else raw
            f1 = {}
            for nm, lay in (("raw", raw), ("refined", refined)):
                _, pm = greedy_match(g, [(b[0],) + _to_xyxy_px(b, W, H) + (b[5],) for b in lay])
                f1[nm] = 2 * sum(pm) / max(len(g) + len(lay), 1)
            rec[kind] = {"dqs_chose_refined": chose, "n_raw": len(raw), "n_refined": len(refined),
                         "f1_raw": f1["raw"], "f1_refined": f1["refined"]}
        per_image.append(rec)

    def to_dets(lay):
        d = {}
        for iid, boxes in lay.items():
            W, H = sizes[iid]
            a_ = np.array([[b[0], (b[1] - b[3] / 2) * W, (b[2] - b[4] / 2) * H, b[3] * W, b[4] * H, b[5]]
                           for b in boxes], float).reshape(-1, 6)
            d[iid] = {"cls": a_[:, 0].astype(int), "boxes": a_[:, 1:5], "score": a_[:, 5]}
        return d

    ap = {k: Scorer(gt, to_dets(v), NC, areas=("all",)).summary()["mAP50"] for k, v in layouts.items()}
    summary = {}
    for kind in ("crop", "inplace"):
        chosen = [r[kind] for r in per_image if r[kind]["dqs_chose_refined"]]
        summary[kind] = {
            "images": len(per_image), "dqs_chose_refined": len(chosen),
            "p_refined_worse_given_chosen": (float(np.mean([c["f1_refined"] < c["f1_raw"] for c in chosen]))
                                             if chosen else None),
            "p_refined_better_given_chosen": (float(np.mean([c["f1_refined"] > c["f1_raw"] for c in chosen]))
                                              if chosen else None),
            "boxes_raw": int(sum(r[kind]["n_raw"] for r in per_image)),
            "boxes_refined": int(sum(r[kind]["n_refined"] for r in per_image))}
    out = {"weights": a.weights, "gt": str(a.gt), "teacher_conf": a.conf, "band": band,
           "layout_mAP50": ap, "summary": summary,
           "note": "per-image quality = F1 at IoU 0.5 of the layout against GT", "per_image": per_image}
    save_json(out, a.out / f"selection_{a.tag}.json")
    log(f"layout mAP50: {ap}")
    log(f"wrote {a.out / f'selection_{a.tag}.json'}")


# ---------------------------------------------------------------------------
def study_calibration(a):
    from ultralytics import YOLO
    gt, files, _ = load_gt(a.gt)
    imgs = resolve_images(files, a.img_root)
    out = {"gt": str(a.gt), "min_conf": a.conf, "models": {}}
    for spec in a.models:
        label, w = spec.split("=", 1)
        model = YOLO(w)
        cmap = class_map_for(model)
        confs, correct = [], []
        for iid, path in tqdm(imgs.items(), desc=f"calibration {label}"):
            img = imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            p = [(b[0],) + _to_xyxy_px(b, W, H) + (b[5],) for b in predict(model, img, a.conf, cmap)]
            _, pm = greedy_match(_gt_xyxy(gt[iid]), p)
            confs += [x[5] for x in p]
            correct += pm
        ece, bins = expected_calibration_error(confs, correct, 15)
        out["models"][label] = {"weights": w, "ece_15": ece, "n_detections": len(confs), "bins": bins}
        log(f"{label}: ECE {ece:.4f} over {len(confs)} detections")
    save_json(out, a.out / f"calibration_{a.tag}.json")


# ---------------------------------------------------------------------------
def study_agreement(a):
    from ultralytics import YOLO
    from .train import agreement
    data = a.data.resolve()
    manifest = load_json(data / "manifest.json")
    human = load_json(data / "human_boxes.json")
    names = set((data / "lists" / f"{a.split}.txt").read_text(encoding="utf-8").split())
    stems = sorted(s for s, m in manifest.items() if m["name"] in names)
    if a.max_images:
        stems = stems[:: max(1, len(stems) // a.max_images)][:a.max_images]
    model = YOLO(a.weights)
    cmap = class_map_for(model)
    labels = {}
    for s in tqdm(stems, desc="agreement"):
        img = imread(data / "store" / "images" / manifest[s]["name"])
        labels[s] = nms_classwise(predict(model, img, a.conf, cmap)) if img is not None else []
    res = agreement(labels, human, manifest, stems)
    out = {"weights": a.weights, "split": a.split, "conf": a.conf, "images": len(stems), **res}
    save_json(out, a.out / f"agreement_{a.tag}.json")
    log(f"precision {res['precision']}, recall {res['recall']}")
    log(f"wrote {a.out / f'agreement_{a.tag}.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", choices=["scale", "selection", "calibration", "agreement"])
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--models", nargs="*", help="calibration: label=weights ...")
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--img-root", type=Path)
    ap.add_argument("--data", type=Path, default=default_data_dir())
    ap.add_argument("--split", default="dev", choices=["train", "dev"])
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--band", default="0.10,0.65")
    ap.add_argument("--scales", default="1,2,4,8,letterbox")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out", type=Path, default=default_results_dir() / "diagnose")
    a = ap.parse_args()
    if a.weights == "yolov8n.pt":
        from .common import find_base_teacher
        a.weights = find_base_teacher()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.study in ("scale", "selection", "calibration") and not (a.gt and a.img_root):
        ap.error(f"{a.study} needs --gt and --img-root")
    {"scale": study_scale, "selection": study_selection,
     "calibration": study_calibration, "agreement": study_agreement}[a.study](a)


if __name__ == "__main__":
    main()
