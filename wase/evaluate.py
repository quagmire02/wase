"""
Evaluate models on a COCO-format GT and write one JSON; build tables from JSONs.

  # ACDC fog val (development set)
  python -m wase.evaluate --set acdc_fog_val \
      --gt <ACDC>/gt_detection/fog/val/instancesonly_fog_val_gt_detection.json \
      --img-root <ACDC>/rgb_anon/fog/val \
      --models coco=yolov8n.pt "runs/naive/seed*/weights/final.pt" \
               "runs/supervised/seed*/weights/final.pt" \
      --compare supervised:naive coco:naive --margin 0.01

  # corpus dev split (human labels, in-domain)
  python -m wase.evaluate --set corpus_dev --gt data/dev_gt.json \
      --img-root data/store/images --models "runs/*/seed*/weights/final.pt"

  # tables for the paper, from every results/*/metrics.json
  python -m wase.evaluate --report

Headline numbers come from pycocotools itself. Our engine (metrics.py) is
checked against them on every run and then used for the bootstrap. The
bootstrap resamples test images AND (when a condition has several seeds) its
training seeds, so intervals include seed-to-seed variation.

Model spec: `label=glob` or a bare glob. For a bare glob matching
runs/<label>/seed<k>/weights/final.pt the label is <label>; every match of one
label is treated as a seed of that condition.
"""

import argparse
import contextlib
import glob
import hashlib
import io
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .common import (ACDC_TO_OURS, CLASSES, IDENTITY, NC, PROJECT, class_map_for,
                     default_results_dir, get_device, load_json, log, nms_classwise, save_json)
from .metrics import Scorer, bootstrap, delta_stats

PRED_CONF, PRED_IOU, PRED_MAXDET = 0.001, 0.7, 300


# ---------------------------------------------------------------------------
def load_gt(path):
    """COCO json -> (gt dict keyed by image id, {id: file_name}, mapping description)."""
    d = load_json(path)
    ours = d.get("wase_format") == "ours"
    cmap = {c["id"]: c["id"] - 1 for c in d["categories"]} if ours else ACDC_TO_OURS
    files = {im["id"]: im["file_name"] for im in d["images"]}
    rows = defaultdict(list)
    skipped = 0
    for an in d["annotations"]:
        c = cmap.get(an["category_id"])
        if c is None:
            skipped += 1
            continue
        x, y, w, h = an["bbox"]
        t = an.get("transmittance")
        rows[an["image_id"]].append((x, y, w, h, c, an.get("area", w * h), bool(an.get("iscrowd", 0)),
                                     np.nan if t is None else t))
    gt = {}
    for iid in files:
        r = rows.get(iid, [])
        arr = np.array([t[:4] for t in r], float).reshape(-1, 4)
        gt[iid] = {"boxes": arr, "cls": np.array([t[4] for t in r], int),
                   "area": np.array([t[5] for t in r], float),
                   "crowd": np.array([t[6] for t in r], bool),
                   "t": np.array([t[7] for t in r], float)}
    return gt, files, {"format": "ours" if ours else "acdc/cityscapes", "skipped_annotations": skipped}


def resolve_images(files, root):
    root = Path(root)
    index = None
    out = {}
    for iid, fn in files.items():
        p = root / fn
        if not p.exists():
            if index is None:
                index = {q.name: q for q in root.rglob("*") if q.is_file()}
            p = index.get(Path(fn).name)
            if p is None:
                raise FileNotFoundError(f"image for GT entry {fn} not found under {root}")
        out[iid] = p
    return out


def expand_models(specs):
    groups = defaultdict(list)
    for spec in specs:
        label, pat = spec.split("=", 1) if "=" in spec else (None, spec)
        matches = sorted(glob.glob(pat)) or ([pat] if Path(pat).name == pat else [])
        if not matches:
            raise FileNotFoundError(f"no checkpoint matches {pat}")
        for m in matches:
            lab = label
            if lab is None:
                mm = re.search(r"runs[\\/]+([^\\/]+)[\\/]+seed\d+", m)
                lab = mm.group(1) if mm else Path(m).stem
            groups[lab].append(m)
    return dict(groups)


def predict_all(weights, images, imgsz, cache_dir):
    from ultralytics import YOLO
    p = Path(weights)
    tag = hashlib.md5(f"{p.resolve() if p.exists() else weights}|"
                      f"{p.stat().st_mtime if p.exists() else ''}|{imgsz}|remap-nms".encode()).hexdigest()[:12]
    cache = cache_dir / f"{p.parent.parent.parent.name}_{p.parent.parent.name}_{p.stem}_{tag}.json" \
        if p.exists() and len(p.parts) > 3 else cache_dir / f"{p.stem}_{tag}.json"
    if cache.exists():
        raw = load_json(cache)
    else:
        model = YOLO(weights)
        cmap = class_map_for(model)
        raw = {}
        for iid, path in tqdm(images.items(), desc=f"predict {p.name}", leave=False):
            r = model.predict(str(path), conf=PRED_CONF, iou=PRED_IOU, imgsz=imgsz,
                              max_det=PRED_MAXDET, verbose=False, device=get_device())[0]
            rows = []
            if r.boxes is not None and len(r.boxes):
                for c, (x1, y1, x2, y2), s in zip(r.boxes.cls.cpu().numpy().astype(int),
                                                   r.boxes.xyxy.cpu().numpy(),
                                                   r.boxes.conf.cpu().numpy()):
                    if c in cmap:
                        rows.append([cmap[c], float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(s)])
            if cmap is not IDENTITY:
                # An 80-class model suppresses truck and car separately; after the
                # remap both are 'car', so one object can carry two boxes. Students
                # (5-class) never have this, so suppress again in our class space.
                rows = _nms_xywh(rows, PRED_IOU)
            raw[str(iid)] = rows
        save_json(raw, cache)
    dets = {}
    for iid, rows in raw.items():
        a = np.array(rows, float).reshape(-1, 6)
        dets[int(iid)] = {"cls": a[:, 0].astype(int), "boxes": a[:, 1:5], "score": a[:, 5]}
    return dets


def _nms_xywh(rows, iou):
    """Class-aware NMS for [cls, x, y, w, h, score] pixel boxes (top-left xywh)."""
    cx = [[r[0], r[1] + r[3] / 2, r[2] + r[4] / 2, r[3], r[4], r[5]] for r in rows]
    return [[b[0], b[1] - b[3] / 2, b[2] - b[4] / 2, b[3], b[4], b[5]] for b in nms_classwise(cx, iou)]


def pycoco_summary(gt, dets):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images = [{"id": int(i)} for i in gt]
    anns, aid = [], 1
    for i, g in gt.items():
        for b, c, ar, cr in zip(g["boxes"], g["cls"], g["area"], g["crowd"]):
            anns.append({"id": aid, "image_id": int(i), "category_id": int(c) + 1,
                         "bbox": [float(v) for v in b], "area": float(ar), "iscrowd": int(cr)})
            aid += 1
    res = [{"image_id": int(i), "category_id": int(c) + 1, "bbox": [float(v) for v in b],
            "score": float(s)} for i, d in dets.items()
           for b, c, s in zip(d["boxes"], d["cls"], d["score"])]
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = {"images": images, "annotations": anns,
                      "categories": [{"id": k + 1, "name": n} for k, n in enumerate(CLASSES)]}
        cg.createIndex()
        cd = cg.loadRes(res) if res else COCO()
        ev = COCOeval(cg, cd, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = [None if v < 0 else float(v) for v in ev.stats]
    return {"mAP50_95": s[0], "mAP50": s[1], "mAP75": s[2], "mAP_small": s[3],
            "mAP_medium": s[4], "mAP_large": s[5], "AR100": s[8]}


# ---------------------------------------------------------------------------
def evaluate(a):
    out_dir = (a.results / a.set).resolve()
    gt, files, gt_info = load_gt(a.gt)
    images = resolve_images(files, a.img_root)
    n_obj = int(sum(len(g["cls"]) for g in gt.values()))
    log(f"{a.set}: {len(gt)} images, {n_obj} objects ({gt_info})")
    groups = expand_models(a.models)

    per_model, scorers = {}, defaultdict(list)
    worst = 0.0
    for label, paths in groups.items():
        per_model[label] = {"checkpoints": paths, "seeds": []}
        for w in paths:
            dets = predict_all(w, images, a.imgsz, out_dir / "preds")
            sc = Scorer(gt, dets, NC)
            ours = sc.summary()
            official = pycoco_summary(gt, dets)
            for k, v in official.items():
                if v is not None and ours.get(k) is not None:
                    worst = max(worst, abs(v - ours[k]))
            per_model[label]["seeds"].append({"checkpoint": w, "pycocotools": official, "ours": ours})
            scorers[label].append(sc)
        for k in ("mAP50", "mAP75", "mAP50_95", "mAP_small", "mAP_medium", "mAP_large"):
            v = [s["pycocotools"][k] for s in per_model[label]["seeds"] if s["pycocotools"][k] is not None]
            per_model[label][k] = {"mean": float(np.mean(v)) if v else None,
                                   "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                   "n_seeds": len(v)}
        iw = [s["ours"]["instance_weighted_AP50"] for s in per_model[label]["seeds"]]
        per_model[label]["instance_weighted_AP50"] = float(np.mean(iw))
        log(f"  {label:<28} mAP50 {per_model[label]['mAP50']['mean']:.4f} "
            f"({len(paths)} seed{'s' if len(paths) > 1 else ''})")
    if worst > 1e-6:
        log(f"[warn] our AP engine differs from pycocotools by up to {worst:.2e}")

    boot = {}
    if a.bootstrap:
        log(f"bootstrap: {a.bootstrap} image resamples, training seeds "
            + ("held fixed" if a.fixed_seeds else "resampled too"))
        draws = bootstrap(scorers, a.bootstrap, a.boot_seed, resample_seeds=not a.fixed_seeds)
        boot["per_model"] = {k: {"mean": float(np.nanmean(v)),
                                 "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
                             for k, v in draws.items()}
        boot["deltas"] = {}
        for pair in a.compare or []:
            x, y = pair.split(":")
            if x not in draws or y not in draws:
                raise SystemExit(f"--compare {pair}: unknown label (have {sorted(draws)})")
            dd = delta_stats(draws[x], draws[y], margin=a.margin)
            dd["observed"] = per_model[y]["mAP50"]["mean"] - per_model[x]["mAP50"]["mean"]
            boot["deltas"][f"{y} - {x}"] = dd
            log(f"  {y} - {x}: {dd['mean']:+.4f} CI95 [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]"
                + (f"  equivalent within ±{a.margin}: {dd['tost']['equivalent']}" if a.margin else ""))

    first = next(iter(scorers.values()))[0]
    result = {"set": a.set, "gt": str(a.gt), "n_images": len(gt), "n_objects": n_obj,
              "objects_per_class": dict(zip(CLASSES, first.summary()["per_class_gt"])),
              "gt_info": gt_info, "imgsz": a.imgsz,
              "prediction": {"conf": PRED_CONF, "iou": PRED_IOU, "max_det": PRED_MAXDET},
              "engine_vs_pycocotools_max_abs_diff": worst,
              "models": per_model, "bootstrap": boot,
              "bootstrap_settings": {"n": a.bootstrap, "seed": a.boot_seed, "unit": "image",
                                     "seeds": ("fixed, averaged within each resample" if a.fixed_seeds else
                                               "resampled with replacement per condition, then averaged")}}
    save_json(result, out_dir / "metrics.json")
    log(f"wrote {out_dir / 'metrics.json'}")


# ---------------------------------------------------------------------------
def fmt(v, nd=3):
    return "--" if v is None else f"{v:.{nd}f}"


def report(results):
    files = sorted(results.glob("*/metrics.json"))
    if not files:
        raise SystemExit(f"no metrics.json under {results}")
    md, tex = ["# Results (generated by `python -m wase.evaluate --report`; do not edit)\n"], []
    for f in files:
        r = load_json(f)
        md.append(f"\n## {r['set']}  ({r['n_images']} images, {r['n_objects']} objects)\n")
        md.append("Objects per class: " + ", ".join(f"{k} {v}" for k, v in r["objects_per_class"].items()) + "\n")
        md.append("Observed values are seed means of pycocotools results; "
                  "CI = 95% paired image bootstrap of the seed-averaged mAP50 "
                  f"(training seeds: {r.get('bootstrap_settings', {}).get('seeds', 'n/a')}).\n")
        md.append("| condition | seeds | mAP50 | (bootstrap CI) | mAP75 | mAP50:95 | small | medium | large | inst.-wt AP50 |")
        md.append("|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
        tex.append(f"% {r['set']}\n\\begin{{tabular}}{{lcccccccc}}\\toprule\n"
                   "Condition & Seeds & AP$_{50}$ & AP$_{75}$ & AP & AP$_S$ & AP$_M$ & AP$_L$ \\\\ \\midrule")
        bpm = r.get("bootstrap", {}).get("per_model", {})
        for lab, m in r["models"].items():
            ci = bpm.get(lab, {}).get("ci95")
            sd = m["mAP50"]["std"]
            md.append(f"| {lab} | {m['mAP50']['n_seeds']} | {fmt(m['mAP50']['mean'])}"
                      f"{'' if sd is None else ' ± ' + fmt(sd)} | "
                      f"{'' if not ci else '[' + fmt(ci[0]) + ', ' + fmt(ci[1]) + ']'} | "
                      f"{fmt(m['mAP75']['mean'])} | {fmt(m['mAP50_95']['mean'])} | "
                      f"{fmt(m['mAP_small']['mean'])} | {fmt(m['mAP_medium']['mean'])} | "
                      f"{fmt(m['mAP_large']['mean'])} | {fmt(m['instance_weighted_AP50'])} |")
            tex.append(f"{lab.replace('_', chr(92) + '_')} & {m['mAP50']['n_seeds']} & {fmt(m['mAP50']['mean'])} & "
                       f"{fmt(m['mAP75']['mean'])} & {fmt(m['mAP50_95']['mean'])} & {fmt(m['mAP_small']['mean'])} & "
                       f"{fmt(m['mAP_medium']['mean'])} & {fmt(m['mAP_large']['mean'])} \\\\")
        tex.append("\\bottomrule\\end{tabular}\n")
        deltas = r.get("bootstrap", {}).get("deltas", {})
        if deltas:
            md.append("\n| comparison (mAP50) | observed Δ | bootstrap mean Δ | 95% CI | CI half-width | p | equivalence (90% CI within ±margin) |")
            md.append("|---|:-:|:-:|:-:|:-:|:-:|:-:|")
            for k, d in deltas.items():
                eq = d.get("tost")
                eq_txt = "--" if not eq else f"{'yes' if eq['equivalent'] else 'no'} (±{eq['margin']})"
                md.append(f"| {k} | {d['observed']:+.4f} | {d['mean']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                          f"{(d['ci95'][1] - d['ci95'][0]) / 2:.4f} | "
                          f"{d['p_two_sided']:.3f} | {eq_txt} |")
    md.append("\nCI half-width is the smallest difference this test set can resolve for that pair. "
              "An equivalence margin smaller than it can only ever return 'inconclusive'.")
    (results / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (results / "tables.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    log(f"wrote {results / 'tables.md'} and {results / 'tables.tex'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", help="name of this evaluation set (results/<set>/)")
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--img-root", type=Path)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--compare", nargs="*", help="pairs a:b, reported as b - a")
    ap.add_argument("--margin", type=float, default=None, help="equivalence margin in AP (e.g. 0.01)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--fixed-seeds", action="store_true",
                    help="do not resample training seeds in the bootstrap (previous round's behaviour)")
    ap.add_argument("--results", type=Path, default=default_results_dir())
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(a.results.resolve())
        return
    if not (a.set and a.gt and a.img_root and a.models):
        ap.error("--set, --gt, --img-root and --models are required (or use --report)")
    evaluate(a)


if __name__ == "__main__":
    main()
