"""
Train any condition, any seed.

  python -m wase.train --condition naive --seed 0
  python -m wase.train --condition crop-dqs --seed 1
  python -m wase.train --condition supervised --seed 0
  python -m wase.train --condition supervised --label-filter visibility --tau 0.1
  python -m wase.train --condition supervised --source rtts
  python -m wase.train --condition supervised --fraction 0.25
  python -m wase.train --condition supervised --imgsz 1024
  python -m wase.train --condition naive --generations 1 --epochs 120
  python -m wase.train --condition verify-flip                  # agreement instead of confidence
  python -m wase.train --condition oracle-addfn                 # naive labels + the objects it missed
  python -m wase.train --condition naive --teacher yolov8m.pt --label-imgsz 1280 \
      --student yolov8n.pt --generations 1 --epochs 120         # strong teacher at label time only

Output: runs/<run-name>/seed<k>/
  config.json            every argument + software versions + git commit
  gen<g>_labels.json     label statistics (incl. agreement with human labels)
  weights/gen<g>.pt      last.pt of each generation (the next teacher)
  weights/final.pt       the model that gets evaluated
  data/                  this run's own image links and label files

Protocol (identical for every condition; see PLAN.md):
  * same training images: the scene-level 'train' split built by prep.py
  * same schedule: 3 stages, lr 1e-3 / 5e-4 / 1e-4, 50 / 40 / 30 epochs, AdamW,
    wd 5e-4, cosine to lr0*0.1, 640 px, batch 16
  * no early stopping and last.pt is kept: the checkpoint is never chosen by a
    validation score (that score is self-agreement for pseudo-label runs)
  * Ultralytics' validation pass uses human labels on a dev subset, purely for
    monitoring; it never selects anything
  * the student of generation g is promoted directly to teacher g+1;
    no cross-generation weight averaging
"""

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from .common import (CLASSES, NAMES, NC, class_map_for, default_data_dir, default_runs_dir,
                     env_info, find_base_teacher, get_device, imread, link_or_copy, load_json,
                     log, read_labels, save_json, seed_everything, stable_unit, write_labels)
from .labeling import ORACLE_KINDS, PSEUDO_CONDITIONS, VERIFY_MODES, apply_oracle, label_image
from .metrics import greedy_match

# Needed for deterministic cuBLAS kernels; must be set before CUDA initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCHEDULE = [  # teacher confidence, lr0, epochs
    (0.15, 1e-3, 50),
    (0.20, 5e-4, 40),
    (0.25, 1e-4, 30),
]
# oracle-<kind>: naive pseudo-labels with one error type repaired from the human labels
ORACLE_CONDITIONS = {f"oracle-{k}": k for k in ORACLE_KINDS}
CONDITIONS = sorted(PSEUDO_CONDITIONS) + sorted(ORACLE_CONDITIONS) + ["supervised"]
MAX_EMPTY_FRACTION = 0.90
VIS_BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.01)]
WIDTH_BINS = [(0, 12), (12, 32), (32, 1e9)]   # box width in px at 640 training scale


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, default=default_data_dir())
    ap.add_argument("--runs", type=Path, default=default_runs_dir())
    ap.add_argument("--name", default=None, help="run name (default: derived from options)")
    ap.add_argument("--teacher", default=None, help="Gen-0 teacher (default: COCO yolov8n.pt)")
    ap.add_argument("--student", default=None,
                    help="Gen-0 student initialisation (default: the Gen-0 teacher). Set it to "
                         "yolov8n.pt when --teacher is a larger model, so the deployed model stays small")
    ap.add_argument("--label-imgsz", type=int, default=None,
                    help="inference size for pseudo-labelling (default: --imgsz). Later generations "
                         "label with the student at this size too")
    ap.add_argument("--verify-mode", default="both", choices=VERIFY_MODES,
                    help="verify-* conditions: admit and reject (both), or only one half")
    ap.add_argument("--oracle-iou", type=float, default=0.5,
                    help="oracle-* conditions: IoU at which a pseudo box matches a human box")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--epochs", default=None,
                    help="comma list per generation (e.g. 50,40,30) or one value for all")
    ap.add_argument("--epochs-scale", type=float, default=1.0, help="multiply epochs (smoke tests)")
    ap.add_argument("--teacher-conf", default=None,
                    help="comma list per generation (default 0.15,0.20,0.25) or one value for all")
    ap.add_argument("--lr-scale", type=float, default=1.0, help="multiply every lr0 (stay-close control)")
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--band", default="0.10,0.65", help="uncertain-confidence band for refinement")
    ap.add_argument("--oversample", action="store_true",
                    help="duplicate images where DQS chose the refined layout (off by default)")
    # supervised options
    ap.add_argument("--label-filter", default="none", choices=["none", "size", "visibility", "teacher"])
    ap.add_argument("--min-px", type=float, default=12.0, help="size filter: min box width/height at 640")
    ap.add_argument("--tau", type=float, default=0.1, help="visibility filter: min transmittance")
    # data options (all conditions)
    ap.add_argument("--source", default="all", choices=["all", "rtts", "city"])
    ap.add_argument("--fraction", type=float, default=1.0, help="nested scene-level subset of train")
    ap.add_argument("--max-images", type=int, default=None, help="cap on train images (smoke tests)")
    ap.add_argument("--val-max", type=int, default=500, help="dev images used for monitoring")
    return ap.parse_args(argv)


def run_name(a):
    if a.name:
        return a.name
    parts = [a.condition]
    if a.label_filter != "none":
        parts.append({"size": f"size{a.min_px:g}", "visibility": f"vis{a.tau:g}",
                      "teacher": "teacher"}[a.label_filter])
    if a.source != "all":
        parts.append(a.source)
    if a.fraction < 1:
        parts.append(f"frac{a.fraction:g}")
    if a.imgsz != 640:
        parts.append(f"img{a.imgsz}")
    if a.lr_scale != 1:
        parts.append(f"lr{a.lr_scale:g}")
    if a.freeze:
        parts.append(f"freeze{a.freeze}")
    if a.teacher:
        parts.append("T" + Path(a.teacher).stem)
    if a.student:
        parts.append("S" + Path(a.student).stem)
    if a.label_imgsz and a.label_imgsz != a.imgsz:
        parts.append(f"label{a.label_imgsz}")
    if a.verify_mode != "both" and a.condition.startswith("verify-"):
        parts.append(a.verify_mode)
    if a.oracle_iou != 0.5 and a.condition in ORACLE_CONDITIONS:
        parts.append(f"oiou{a.oracle_iou:g}")
    if a.generations != 3:
        parts.append(f"{a.generations}gen")
    if a.epochs:
        parts.append(f"ep{a.epochs.replace(',', '-')}")
    if a.teacher_conf and a.condition != "supervised":
        parts.append("conf" + a.teacher_conf.replace(",", "-"))
    if a.oversample:
        parts.append("oversample")
    if a.band != "0.10,0.65" and a.condition in PSEUDO_CONDITIONS:
        parts.append("band" + a.band.replace(",", "-"))
    return "_".join(parts)


def schedule(a):
    sched = [list(s) for s in SCHEDULE[:a.generations]]
    if len(sched) < a.generations:
        sched += [list(SCHEDULE[-1])] * (a.generations - len(sched))
    if a.epochs:
        ep = [int(e) for e in a.epochs.split(",")]
        ep = ep * a.generations if len(ep) == 1 else ep
        if len(ep) != a.generations:
            raise SystemExit("--epochs needs one value or one per generation")
        for s, e in zip(sched, ep):
            s[2] = e
    if a.teacher_conf:
        cf = [float(c) for c in a.teacher_conf.split(",")]
        cf = cf * a.generations if len(cf) == 1 else cf
        if len(cf) != a.generations:
            raise SystemExit("--teacher-conf needs one value or one per generation")
        for s, c in zip(sched, cf):
            s[0] = c
    for s in sched:
        s[1] *= a.lr_scale
        s[2] = max(1, int(round(s[2] * a.epochs_scale)))
    return sched


# ---------------------------------------------------------------------------
def select_images(a, manifest):
    names = (a.data / "lists" / "train.txt").read_text(encoding="utf-8").split()
    by_name = {m["name"]: (stem, m) for stem, m in manifest.items()}
    out = []
    for n in names:
        stem, m = by_name[n]
        if a.source != "all" and m["source"] != a.source:
            continue
        if a.fraction < 1 and stable_unit("frac:" + m["scene"]) >= a.fraction:
            continue
        out.append(stem)
    if a.max_images:
        out = sorted(out, key=lambda s: stable_unit("cap:" + s))[:a.max_images]
    return sorted(out)


def build_view(a, run, manifest, train_stems):
    """Per-run image links (never shared label folders between runs)."""
    store = a.data / "store" / "images"
    tr_img, va_img = run / "data" / "train" / "images", run / "data" / "val" / "images"
    for s in tqdm(train_stems, desc="linking train images", leave=False):
        link_or_copy(store / manifest[s]["name"], tr_img / manifest[s]["name"])
    dev = (a.data / "lists" / "dev.txt").read_text(encoding="utf-8").split()
    dev = sorted(dev, key=lambda n: stable_unit("val:" + n))[:a.val_max]
    by_name = {m["name"]: stem for stem, m in manifest.items()}
    va_lab = run / "data" / "val" / "labels"
    va_lab.mkdir(parents=True, exist_ok=True)
    for n in dev:
        link_or_copy(store / n, va_img / n)
        shutil.copy(a.data / "human_labels" / f"{by_name[n]}.txt", va_lab / f"{Path(n).stem}.txt")
    yml = run / "data" / "dataset.yaml"
    yml.write_text(yaml.safe_dump({"path": str((run / "data").resolve()), "train": "train/images",
                                   "val": "val/images", "nc": NC, "names": NAMES},
                                  sort_keys=False), encoding="utf-8")
    return yml, len(dev)


def reset_labels(run):
    lab = run / "data" / "train" / "labels"
    if lab.exists():
        shutil.rmtree(lab)
    lab.mkdir(parents=True)
    for dup in (run / "data" / "train" / "images").glob("dup_*"):
        dup.unlink()
    for c in (run / "data").rglob("*.cache"):
        c.unlink()
    return lab


# ---------------------------------------------------------------------------
def human_boxes_filtered(a, stems, manifest, human, teacher_cache):
    out, dropped = {}, Counter()
    for s in stems:
        m = manifest[s]
        scale = 640 / max(m["width"], m["height"])
        keep = []
        for b in human[s]:
            c, cx, cy, w, h, t = b
            if a.label_filter == "size" and min(w * m["width"], h * m["height"]) * scale < a.min_px:
                dropped["size"] += 1
                continue
            if a.label_filter == "visibility" and t is not None and t < a.tau:
                dropped["visibility"] += 1
                continue
            if a.label_filter == "teacher":
                tb = teacher_cache.get(s, [])
                g = [(c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)]
                pr = [(p[0], p[1] - p[3] / 2, p[2] - p[4] / 2, p[1] + p[3] / 2, p[2] + p[4] / 2, p[5])
                      for p in tb]
                if not greedy_match(g, pr)[0][0]:
                    dropped["teacher_missed"] += 1
                    continue
            keep.append([c, cx, cy, w, h, 1.0])
        out[s] = keep
    return out, dict(dropped)


def coco_teacher_cache(a, stems, manifest):
    """Raw COCO-teacher predictions (conf 0.15) for the 'teacher' label filter."""
    from ultralytics import YOLO
    p = a.data / "cache" / f"coco_teacher_conf0.15_img{a.imgsz}.json"
    cache = load_json(p) if p.exists() else {}
    todo = [s for s in stems if s not in cache]
    if todo:
        from .common import nms_classwise, predict
        model = YOLO(a.teacher or find_base_teacher())
        cmap = class_map_for(model)
        for s in tqdm(todo, desc="COCO teacher (for label filter)"):
            img = imread(a.data / "store" / "images" / manifest[s]["name"])
            cache[s] = nms_classwise(predict(model, img, 0.15, cmap, imgsz=a.imgsz)) if img is not None else []
        save_json(cache, p)
    return cache


def agreement(labels, human, manifest, stems):
    """Pseudo-labels vs human labels (IoU >= 0.5), by size and visibility bin."""
    tp_p = n_p = 0
    by_w = {f"{lo}-{hi:g}px": [0, 0] for lo, hi in WIDTH_BINS}
    by_v = {f"t{lo}-{hi:g}": [0, 0] for lo, hi in VIS_BINS}
    by_v["t_unknown(rtts)"] = [0, 0]
    by_src = {"rtts": [0, 0], "city": [0, 0]}
    for s in stems:
        m = manifest[s]
        scale = 640 / max(m["width"], m["height"])
        g = [(b[0], b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2) for b in human[s]]
        p = [(b[0], b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2, b[5])
             for b in labels.get(s, [])]
        gm, pm = greedy_match(g, p)
        tp_p += sum(pm)
        n_p += len(pm)
        for b, hit in zip(human[s], gm):
            wpx = b[3] * m["width"] * scale
            for lo, hi in WIDTH_BINS:
                if lo <= wpx < hi:
                    by_w[f"{lo}-{hi:g}px"][0] += hit
                    by_w[f"{lo}-{hi:g}px"][1] += 1
            t = b[5]
            key = "t_unknown(rtts)" if t is None else next(
                f"t{lo}-{hi:g}" for lo, hi in VIS_BINS if lo <= t < hi)
            by_v[key][0] += hit
            by_v[key][1] += 1
            by_src[m["source"]][0] += hit
            by_src[m["source"]][1] += 1

    def rec(d):
        return {k: {"recall": (v[0] / v[1]) if v[1] else None, "n_human": v[1]} for k, v in d.items()}
    n_h = sum(v[1] for v in by_src.values())
    tp_h = sum(v[0] for v in by_src.values())
    r = tp_h / n_h if n_h else None
    pr = tp_p / n_p if n_p else None
    return {"precision": pr, "recall": r,
            "f1": (2 * pr * r / (pr + r)) if pr and r else None,
            "recall_by_width_at_640": rec(by_w), "recall_by_transmittance": rec(by_v),
            "recall_by_source": rec(by_src)}


# ---------------------------------------------------------------------------
def generate_pseudo(a, teacher_path, conf, stems, manifest, spec, band, human=None, oracle=None):
    from ultralytics import YOLO
    model = YOLO(teacher_path)
    cmap = class_map_for(model)
    labels, st = {}, Counter()
    recs, picks = [], []
    import torch
    with torch.inference_mode():
        for s in tqdm(stems, desc=f"pseudo-labels ({Path(str(teacher_path)).name})"):
            img = imread(a.data / "store" / "images" / manifest[s]["name"])
            if img is None:
                st["unreadable"] += 1
                continue
            boxes, info = label_image(model, img, cmap, spec, conf, band,
                                      a.label_imgsz or a.imgsz, a.verify_mode)
            if oracle:
                boxes, oc = apply_oracle(oracle, boxes, human[s], a.oracle_iou)
                st.update(oc)
            for k in ("n_candidates", "n_admitted", "n_rejected"):
                if k in info:
                    st[k] += info[k]
            labels[s] = boxes
            st["boxes"] += len(boxes)
            st["boxes_raw_view"] += info.get("n_raw", 0)
            st["boxes_second_view"] += info.get("n_second", 0)
            st["empty"] += not boxes
            if "dqs_prefers_second" in info:
                st["images_with_second_view"] += 1
                st["dqs_prefers_second"] += info["dqs_prefers_second"]
                picks.append((s, info["dqs_prefers_second"]))
            recs.extend(info.get("records", []))
            for b in boxes:
                st[f"class_{CLASSES[b[0]]}"] += 1
    acc = [r for r in recs if r["accepted"]]
    cand = [r for r in recs if r["conf_best"] is not None]
    refine = {
        "uncertain_boxes": len(recs), "with_any_same_class_repredict": len(cand),
        "accepted": len(acc),
        "mean_dconf_accepted": float(np.mean([r["conf_best"] - r["conf_raw"] for r in acc])) if acc else None,
        "mean_dconf_unconditional": float(np.mean([r["conf_best"] - r["conf_raw"] for r in cand])) if cand else None,
        "median_area_ratio_accepted": float(np.median([r["area_ratio"] for r in acc])) if acc else None,
    } if recs else None
    return labels, dict(st), refine, picks


def train_generation(a, run, g, init_weights, yml, lr0, epochs):
    from ultralytics import YOLO
    model = YOLO(init_weights)
    model.train(data=str(yml), epochs=epochs, imgsz=a.imgsz, batch=a.batch, lr0=lr0, lrf=0.1,
                cos_lr=True, optimizer="AdamW", weight_decay=5e-4, patience=epochs + 1,
                freeze=a.freeze or None, seed=a.seed, deterministic=True, workers=a.workers,
                device=get_device(), project=str(run / "ultralytics"), name=f"gen{g}",
                exist_ok=True, cache=False, plots=False, verbose=False)
    last = run / "ultralytics" / f"gen{g}" / "weights" / "last.pt"
    if not last.exists():
        raise FileNotFoundError(f"training finished but {last} is missing")
    dst = run / "weights" / f"gen{g}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(last, dst)
    probe = YOLO(str(dst))
    if len(probe.names) != NC:
        raise RuntimeError(f"gen {g} student has nc={len(probe.names)}, expected {NC}")
    return str(dst)


def main(argv=None):
    a = parse_args(argv)
    a.data = a.data.resolve()
    a.runs = a.runs.resolve()
    if not (a.data / "manifest.json").exists():
        raise SystemExit(f"{a.data} is not built; run `python -m wase.prep` first")
    name = run_name(a)
    run = a.runs / name / f"seed{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    logf = run / "train.log"
    seed_everything(a.seed)
    get_device()
    band = tuple(float(x) for x in a.band.split(","))
    sched = schedule(a)
    teacher0 = a.teacher or find_base_teacher()
    student0 = a.student or teacher0
    if band[0] >= band[1]:
        raise SystemExit("--band needs lower < upper")

    manifest = load_json(a.data / "manifest.json")
    human = load_json(a.data / "human_boxes.json")
    stems = select_images(a, manifest)
    if not stems:
        raise SystemExit("no training images selected")
    yml, n_val = build_view(a, run, manifest, stems)

    cfg = {"run": name, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
           "schedule": [{"teacher_conf": c, "lr0": lr, "epochs": e} for c, lr, e in sched],
           "n_train_images": len(stems), "n_val_monitor_images": n_val,
           "gen0_teacher": teacher0, "gen0_student_init": student0, "env": env_info()}
    save_json(cfg, run / "config.json")
    log(f"run {name} seed {a.seed}: {len(stems)} train images, {n_val} monitor images", logf)

    teacher = teacher0
    teacher_cache = coco_teacher_cache(a, stems, manifest) if a.label_filter == "teacher" else None
    for g, (conf, lr0, epochs) in enumerate(sched):
        done = run / "weights" / f"gen{g}.pt"
        if done.exists():
            log(f"gen {g} already trained -> {done}", logf)
            teacher = str(done)
            continue
        lab = reset_labels(run)
        if a.condition == "supervised":
            labels, dropped = human_boxes_filtered(a, stems, manifest, human, teacher_cache)
            st = {"boxes": sum(len(v) for v in labels.values()),
                  "empty": sum(not v for v in labels.values()), "dropped_by_filter": dropped}
            refine, picks = None, []
        else:
            oracle = ORACLE_CONDITIONS.get(a.condition)
            spec = PSEUDO_CONDITIONS["naive" if oracle else a.condition]
            log(f"gen {g}: labelling {len(stems)} images with {Path(str(teacher)).name} at conf {conf}", logf)
            labels, st, refine, picks = generate_pseudo(a, teacher, conf, stems, manifest, spec, band,
                                                        human, oracle)
        n = len(stems)
        if st["boxes"] == 0:
            raise RuntimeError(f"gen {g}: zero boxes -- the teacher is dead; refusing to train")
        if st["empty"] / n > MAX_EMPTY_FRACTION:
            raise RuntimeError(f"gen {g}: {100 * st['empty'] / n:.1f}% empty labels; refusing to train")

        oversampled = 0
        for s in stems:
            nm = manifest[s]["name"]
            write_labels(lab / f"{Path(nm).stem}.txt", labels.get(s, []))
        if a.oversample:
            img_dir = run / "data" / "train" / "images"
            for s, chose in picks:
                if chose:
                    nm = manifest[s]["name"]
                    link_or_copy(img_dir / nm, img_dir / f"dup_{nm}")
                    write_labels(lab / f"dup_{Path(nm).stem}.txt", labels[s])
                    oversampled += 1

        stats = {"generation": g, "teacher": str(teacher), "teacher_conf": conf, "images": n,
                 "empty_fraction": st["empty"] / n, "oversampled": oversampled, "label_stats": st,
                 "refinement": refine,
                 "agreement_with_human_labels": agreement(labels, human, manifest, stems)}
        save_json(stats, run / f"gen{g}_labels.json")
        ag = stats["agreement_with_human_labels"]
        log(f"gen {g}: {st['boxes']} boxes, {100 * st['empty'] / n:.1f}% empty, "
            f"vs human P={ag['precision']} R={ag['recall']}", logf)

        init = student0 if g == 0 else teacher
        teacher = train_generation(a, run, g, init, yml, lr0, epochs)
        log(f"gen {g} trained: lr0 {lr0}, {epochs} epochs -> {teacher}", logf)

    final = run / "weights" / "final.pt"
    shutil.copy2(teacher, final)
    log(f"done: {final}", logf)
    return str(final)


if __name__ == "__main__":
    main()
