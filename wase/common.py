"""
Shared pieces: paths, the class table, class maps, image I/O, DCP, box maths,
NMS, weighted boxes fusion, DQS, label I/O, seeding and run metadata.

Every other module imports from here, so a constant lives in exactly one place.
"""

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT.parent


def default_data_dir():
    return Path(os.environ.get("WASE_DATA", PROJECT / "data")).resolve()


def default_runs_dir():
    return Path(os.environ.get("WASE_RUNS", PROJECT / "runs")).resolve()


def default_results_dir():
    return Path(os.environ.get("WASE_RESULTS", PROJECT / "results")).resolve()


def find_source_dataset():
    """Ultimate_Fog_Dataset as a folder or as the original zip."""
    env = os.environ.get("WASE_SOURCE_DATA")
    cands = [Path(env)] if env else []
    cands += [PARENT / "Ultimate_Fog_Dataset", PARENT / "Ultimate_Fog_Dataset.zip",
              PROJECT / "Ultimate_Fog_Dataset", PROJECT / "Ultimate_Fog_Dataset.zip"]
    for c in cands:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        "Ultimate_Fog_Dataset not found (folder or .zip). Looked in:\n  "
        + "\n  ".join(str(c) for c in cands)
        + "\nSet WASE_SOURCE_DATA to its location.")


def find_rtts_xml():
    env = os.environ.get("WASE_RTTS_XML")
    for c in ([Path(env)] if env else []) + [PARENT / "RTTS" / "Annotations",
                                              PROJECT / "RTTS" / "Annotations"]:
        if c.is_dir():
            return c.resolve()
    return None


def find_base_teacher():
    """COCO-pretrained YOLOv8n; Ultralytics downloads it if no local copy exists."""
    for c in (PROJECT / "yolov8n.pt", PARENT / "yolov8n.pt"):
        if c.exists():
            return str(c.resolve())
    return "yolov8n.pt"


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus"]
NC = len(CLASSES)
NAMES = {i: n for i, n in enumerate(CLASSES)}

# COCO-80 detector output -> ours. truck->car, train->bus.
COCO_TO_OURS = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 4, 7: 2}
IDENTITY = {i: i for i in range(NC)}

# Id space actually used by Ultimate_Fog_Dataset's Cityscapes .txt files
# (8-class Cityscapes list without 'train'); rider->person, truck->car.
CITYSCAPES_TXT_TO_OURS = {0: 0, 1: 0, 2: 2, 3: 2, 4: 4, 5: 3, 6: 1}

# RTTS VOC names. Verified 2026-10-21: the shipped .txt ids already match ours,
# but the .txt files keep only ~72% of the XML boxes, so XML is preferred.
RTTS_NAME_TO_OURS = {"person": 0, "bicycle": 1, "car": 2, "motorbike": 3,
                     "motorcycle": 3, "bus": 4}

# ACDC / Cityscapes panoptic category ids -> ours. rider->person, truck->car,
# train->bus.
ACDC_TO_OURS = {24: 0, 25: 0, 26: 2, 27: 2, 28: 4, 31: 4, 32: 3, 33: 1}


def class_map_for(model):
    """Prediction remap from the RUNNING model's class count, never a constant."""
    nc = len(model.names)
    if nc == 80:
        return COCO_TO_OURS
    if nc == NC:
        return IDENTITY
    raise ValueError(f"model has nc={nc}; expected 80 (COCO) or {NC}")


# ---------------------------------------------------------------------------
# Image I/O and hashing
# ---------------------------------------------------------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def imread(path):
    """cv2.imread returns None for non-ASCII Windows paths; decode in memory."""
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except Exception:
        return None


def stable_unit(key):
    """Deterministic number in [0, 1) from a string (used for splits/subsets)."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10_000_000 / 10_000_000


def link_or_copy(src, dst):
    dst = Path(dst)
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Dark Channel Prior (He et al., TPAMI 2011) -- the formulation under test.
# Kept identical to the previous round so results are comparable; its known
# departures from He et al. (percentile airlight, no transmission refinement,
# ROI-local airlight for crops) are documented in PLAN.md.
# ---------------------------------------------------------------------------
DCP_OMEGA = 0.95
DCP_T_MIN = 0.25
DCP_KERNEL_ROI = 3
DCP_KERNEL_FULL = 15


def dehaze(img, kernel=DCP_KERNEL_ROI, omega=DCP_OMEGA, t_min=DCP_T_MIN):
    if img is None or img.size == 0:
        return img
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    atmo = np.percentile(img, 99, axis=(0, 1))
    norm = img.astype(np.float32) / (atmo.astype(np.float32) + 1e-6)
    t = np.maximum(1.0 - omega * cv2.erode(norm.min(axis=2), k), t_min)
    out = (img.astype(np.float32) - atmo) / t[..., None] + atmo
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Boxes. Internal format: [cls, cx, cy, w, h, conf], coordinates normalised.
# ---------------------------------------------------------------------------
def cxcywh_to_xyxy(b):
    return (b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2)


def iou_xyxy(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def iou_cxcywh(a, b):
    return iou_xyxy(cxcywh_to_xyxy(a), cxcywh_to_xyxy(b))


def nms_classwise(boxes, iou_thr=0.5):
    """Post-remap, class-aware NMS (truck and car both become 'car')."""
    boxes = sorted(boxes, key=lambda b: -b[5])
    keep = []
    while boxes:
        cur = boxes.pop(0)
        keep.append(cur)
        boxes = [o for o in boxes
                 if o[0] != cur[0] or iou_cxcywh(cur[1:5], o[1:5]) < iou_thr]
    return keep


def dqs(boxes):
    """Detection Quality Score of the previous round: mean confidence * sqrt(N)."""
    if not boxes:
        return 0.0
    return float(np.mean([b[5] for b in boxes]) * np.sqrt(len(boxes)))


def weighted_boxes_fusion(views, iou_thr=0.55):
    """
    WBF (Solovyev et al., 2021) over a list of views, each a list of boxes.
    Fused score = mean member score * min(#views seen, n_views) / n_views.
    """
    n_views = len(views)
    tagged = [(b, v) for v, boxes in enumerate(views) for b in boxes]
    tagged.sort(key=lambda t: -t[0][5])
    clusters = []  # [members, view_ids, fused]
    for b, v in tagged:
        hit = None
        for c in clusters:
            if c[2][0] == b[0] and iou_cxcywh(c[2][1:5], b[1:5]) > iou_thr:
                hit = c
                break
        if hit is None:
            clusters.append([[b], {v}, list(b)])
            continue
        hit[0].append(b)
        hit[1].add(v)
        w = np.array([m[5] for m in hit[0]], dtype=float)
        for j in range(1, 5):
            hit[2][j] = float(np.average([m[j] for m in hit[0]], weights=w))
        hit[2][5] = float(w.mean())
    out = []
    for members, vids, fused in clusters:
        fused[5] *= min(len(vids), n_views) / n_views
        out.append(fused)
    return out


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(model, img, conf, cmap, imgsz=640, iou=0.5, device=None, max_det=300):
    """Run a detector on one BGR image; return boxes in our class space."""
    r = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, verbose=False,
                      device=device if device is not None else get_device(),
                      max_det=max_det)[0]
    out = []
    if r.boxes is not None and len(r.boxes):
        cls = r.boxes.cls.cpu().numpy().astype(int)
        xywhn = r.boxes.xywhn.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        for c, (x, y, w, h), s in zip(cls, xywhn, cf):
            if c in cmap:
                out.append([cmap[c], float(x), float(y), float(w), float(h), float(s)])
    return out


# ---------------------------------------------------------------------------
# Label I/O (YOLO txt)
# ---------------------------------------------------------------------------
def write_labels(path, boxes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def read_labels(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.split()
        if len(s) >= 5:
            out.append([int(float(s[0]))] + [float(v) for v in s[1:5]] + [1.0])
    return out


def clamp_box(cx, cy, w, h):
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
_DEVICE = None


def get_device():
    """GPU 0 if a CUDA kernel really executes, else CPU (with a loud warning)."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    import torch
    if torch.cuda.is_available():
        try:
            (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).sum().item()
            _DEVICE = 0
            return _DEVICE
        except Exception as exc:
            print(f"[warn] CUDA reported available but no kernel ran ({exc}); "
                  f"this torch build probably does not support this GPU.")
    print("[warn] running on CPU -- training will be very slow")
    _DEVICE = "cpu"
    return _DEVICE


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_info():
    import torch
    import ultralytics
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        info["git_commit"] = subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], capture_output=True,
            text=True, timeout=10).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(PROJECT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        info["git_dirty"] = bool(dirty)
    except Exception:
        info["git_commit"] = None
    return info


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def log(msg, path=None):
    from datetime import datetime
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(line.encode(enc, "replace").decode(enc), flush=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
