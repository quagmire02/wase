"""
Label-free baseline: adapt only the BatchNorm statistics to fog (AdaBN,
Li et al. 2018). No labels, no gradient step, no weight changes.

  python -m wase.adabn                       # -> runs/adabn/seed0/weights/final.pt
  python -m wase.adabn --max-images 500 --seed 1

Why it exists: PLAN.md credits part of naive self-training's gain to the
student's normalisation layers adapting to foggy inputs. This run measures how
much of the gain that alone buys. The teacher's running mean/variance are
replaced by the statistics of the foggy training images; everything else is the
COCO model, so the output keeps its 80 classes (evaluate.py handles that).

The seed only chooses which training images are used when --max-images is set.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .common import (default_data_dir, default_runs_dir, env_info, find_base_teacher, get_device,
                     imread, load_json, log, save_json, seed_everything, stable_unit)


def adapt_batchnorm(weights, paths, imgsz=640, batch=16):
    """Returns (YOLO model with fog BatchNorm statistics, number of images used)."""
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    yolo = YOLO(weights)
    dev = get_device()
    dev = torch.device("cpu" if dev == "cpu" else f"cuda:{dev}")
    net = yolo.model.to(dev).float().eval()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise RuntimeError("model has no BatchNorm layers (already fused?)")
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None      # cumulative average over everything it sees
        m.train()
    box = LetterBox((imgsz, imgsz), auto=False)
    used, buf = 0, []

    def flush():
        x = torch.from_numpy(np.stack(buf)).to(dev).float() / 255.0
        net(x)
        buf.clear()

    with torch.no_grad():
        for p in tqdm(paths, desc="BatchNorm statistics"):
            img = imread(p)
            if img is None:
                continue
            buf.append(np.ascontiguousarray(box(image=img)[..., ::-1].transpose(2, 0, 1)))
            used += 1
            if len(buf) == batch:
                flush()
        if len(buf) > 1:       # a batch of one gives a degenerate variance
            flush()
    for m, mom in zip(bns, momenta):
        m.momentum = mom
        m.eval()
    return yolo, used


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=default_data_dir())
    ap.add_argument("--runs", type=Path, default=default_runs_dir())
    ap.add_argument("--name", default="adabn")
    ap.add_argument("--teacher", default=None, help="model to adapt (default: COCO yolov8n.pt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-images", type=int, default=None, help="default: every training image")
    a = ap.parse_args(argv)
    data = a.data.resolve()
    if not (data / "manifest.json").exists():
        raise SystemExit(f"{data} is not built; run `python -m wase.prep` first")
    seed_everything(a.seed)
    names = (data / "lists" / "train.txt").read_text(encoding="utf-8").split()
    if a.max_images:
        names = sorted(names, key=lambda n: stable_unit(f"adabn{a.seed}:" + n))[:a.max_images]
    run = a.runs.resolve() / a.name / f"seed{a.seed}"
    weights = a.teacher or find_base_teacher()
    model, used = adapt_batchnorm(weights, [data / "store" / "images" / n for n in sorted(names)],
                                  a.imgsz, a.batch)
    out = run / "weights" / "final.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("final.tmp.pt")
    model.save(str(tmp))
    shutil.move(str(tmp), str(out))
    save_json({"run": a.name, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
               "source_model": weights, "n_images_used": used, "env": env_info()}, run / "config.json")
    log(f"BatchNorm statistics from {used} images -> {out}", run / "train.log")
    return str(out)


if __name__ == "__main__":
    main()
