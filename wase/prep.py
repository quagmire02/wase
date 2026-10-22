"""
One-time data build.

  python -m wase.prep                    # full build
  python -m wase.prep --limit-scenes 40  # tiny build for a smoke test

Reads Ultimate_Fog_Dataset (folder or the original .zip, never modified) and
RTTS VOC XML, and writes everything into <data>/:

  store/images/<file>        one copy of every usable image
  human_labels/<stem>.txt    human labels, our 5 classes, YOLO format
  human_boxes.json           {stem: [[cls, cx, cy, w, h, transmittance|None], ...]}
  manifest.json              one record per image (source, scene, beta, split, size)
  lists/train.txt, dev.txt   image file names per split
  dev_gt.json                COCO-format GT of the dev split (for evaluate/diagnose)
  audit.json                 every count the paper cites

Fixes relative to the previous round
------------------------------------
* Labels are looked up by SCENE across both source folders. The old lookup
  only searched the image's own folder and silently lost 1,918 labelled images.
* The split is by scene, so the three fog renders of one street never straddle
  train and dev.
* The 10 BBOX_* overlay renders and the 30 images with no label anywhere are
  excluded from every condition, so all conditions see the same images.
* RTTS uses the VOC XML (the shipped .txt files keep only ~72% of the boxes).
* Per-box transmittance is recovered for Foggy Cityscapes from its three fog
  densities (see visibility_for_scene).
"""

import argparse
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .common import (CITYSCAPES_TXT_TO_OURS, CLASSES, IMG_EXTS, NC, RTTS_NAME_TO_OURS,
                     clamp_box, default_data_dir, find_rtts_xml, find_source_dataset,
                     log, save_json, stable_unit, write_labels)

DEV_FRACTION = 0.10
MIN_BOX_PX = 4
BETAS = (0.005, 0.01, 0.02)
CITY_MARK = "_leftImg8bit"
BETA_RE = re.compile(r"_foggy_beta_([\d.]+)$")


# ---------------------------------------------------------------------------
# Source access: folder or zip, same interface
# ---------------------------------------------------------------------------
class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path) if self.path.suffix.lower() == ".zip" else None

    def files(self):
        """Yield (key, parts) where parts are path components below the dataset root."""
        if self.zip:
            for n in self.zip.namelist():
                if not n.endswith("/"):
                    parts = n.split("/")
                    if parts[0].lower().startswith("ultimate_fog_dataset"):
                        parts = parts[1:]
                    yield n, parts
        else:
            for p in self.path.rglob("*"):
                if p.is_file():
                    yield str(p), list(p.relative_to(self.path).parts)

    def read_bytes(self, key):
        return self.zip.read(key) if self.zip else Path(key).read_bytes()

    def size(self, key):
        return self.zip.getinfo(key).file_size if self.zip else Path(key).stat().st_size

    def copy_to(self, key, dst):
        dst = Path(dst)
        if dst.exists() and dst.stat().st_size == self.size(key):
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self.zip:
            with self.zip.open(key) as src, open(dst, "wb") as out:
                while True:
                    chunk = src.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
        else:
            import shutil
            shutil.copy2(key, dst)


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------
def scene_of(stem):
    return BETA_RE.sub("", stem)


def beta_of(stem):
    m = BETA_RE.search(stem)
    return float(m.group(1)) if m else None


def parse_city_txt(text, W, H, stats):
    out = []
    for line in text.splitlines():
        s = line.split()
        if len(s) < 5:
            continue
        cls = CITYSCAPES_TXT_TO_OURS.get(int(float(s[0])))
        if cls is None:
            stats["city_unmapped_id"] += 1
            continue
        cx, cy, w, h = (float(v) for v in s[1:5])
        if cx - w / 2 < 0 or cy - h / 2 < 0 or cx + w / 2 > 1 or cy + h / 2 > 1:
            stats["boxes_clamped"] += 1
        cx, cy, w, h = clamp_box(cx, cy, w, h)
        if w * W < MIN_BOX_PX or h * H < MIN_BOX_PX:
            stats["boxes_below_min_px"] += 1
            continue
        out.append([cls, cx, cy, w, h])
    return out


def parse_voc(path, stats):
    root = ET.parse(path).getroot()
    W, H = int(root.find("size/width").text), int(root.find("size/height").text)
    out = []
    for o in root.findall("object"):
        name = (o.find("name").text or "").strip().lower()
        cls = RTTS_NAME_TO_OURS.get(name)
        if cls is None:
            stats["rtts_unmapped_name"] += 1
            continue
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
        if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
            stats["boxes_below_min_px"] += 1
            continue
        out.append([cls, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H])
    return out, (W, H)


def dedupe(boxes, stats):
    seen, out = set(), []
    for b in boxes:
        key = (b[0],) + tuple(round(v, 5) for v in b[1:5])
        if key in seen:
            stats["duplicate_boxes_removed"] += 1
            continue
        seen.add(key)
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# Visibility: Foggy Cityscapes renders I_b = J t_b + A (1 - t_b), t_b = exp(-b d).
# Betas 0.005/0.01/0.02 double, so t, t^2, t^4 for one t, which gives
#   (I3 - A)(I1 - A)^2 = (I2 - A)^3
# -> airlight A, then t = (I2 - A) / (I1 - A) per pixel.
# ---------------------------------------------------------------------------
_A_GRID = np.arange(100, 256, 0.5)


def visibility_for_scene(I1, I2, I3, rng, n_sample=20000):
    """Return (t_map at beta=0.005, airlight, fit residual) from luminance images."""
    idx = rng.choice(I1.size, min(n_sample, I1.size), replace=False)
    a1, a2, a3 = I1.flat[idx], I2.flat[idx], I3.flat[idx]
    best_cost, best_A = np.inf, None
    for A in _A_GRID:
        cost = np.median(np.abs((a3 - A) * (a1 - A) ** 2 - (a2 - A) ** 3)
                         / np.maximum(np.abs(a2 - A) ** 3, 1.0))
        if cost < best_cost:
            best_cost, best_A = cost, A
    d1 = I1 - best_A
    ok = np.abs(d1) > 4
    t = np.full(I1.shape, np.nan)
    t[ok] = np.clip((I2[ok] - best_A) / d1[ok], 0.0, 1.0)
    return t, float(best_A), float(best_cost)


def box_transmittance(t_map, box):
    H, W = t_map.shape
    _, cx, cy, w, h = box[:5]
    x1, x2 = int(max(0, (cx - w / 2) * W)), int(min(W, (cx + w / 2) * W))
    y1, y2 = int(max(0, (cy - h / 2) * H)), int(min(H, (cy + h / 2) * H))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    v = np.nanmedian(t_map[y1:y2, x1:x2]) if np.isfinite(t_map[y1:y2, x1:x2]).any() else np.nan
    return None if not np.isfinite(v) else float(v)


def lum(path):
    img = cv2.imread(str(path))
    return None if img is None else img.astype(np.float64).mean(axis=2)


# ---------------------------------------------------------------------------
def build(data, limit_scenes=None, visibility=True, max_fit_residual=0.05):
    src_path = find_source_dataset()
    rtts_xml = find_rtts_xml()
    log(f"source dataset : {src_path}")
    log(f"RTTS VOC XML   : {rtts_xml or 'NOT FOUND -> falling back to RTTS .txt'}")
    log(f"output         : {data}")
    src = Source(src_path)
    stats = Counter()

    images, label_txt = {}, {}
    for key, parts in src.files():
        name = parts[-1]
        ext = Path(name).suffix.lower()
        folder = parts[0] if len(parts) > 1 else ""
        stem = Path(name).stem
        if ext in IMG_EXTS:
            if name.startswith("BBOX_"):
                stats["excluded_bbox_overlay"] += 1
                continue
            if stem in images:
                raise RuntimeError(f"duplicate image stem across folders: {stem}")
            images[stem] = {"key": key, "name": name, "orig_folder": folder}
        elif ext == ".txt":
            label_txt.setdefault(stem, []).append(key)
    log(f"found {len(images)} images, {len(label_txt)} label files")

    city_label_by_scene = {}
    for stem, keys in label_txt.items():
        if CITY_MARK in stem:
            city_label_by_scene.setdefault(scene_of(stem), keys[0])

    if limit_scenes:
        city_sc = sorted({scene_of(s) for s in images if CITY_MARK in s},
                         key=lambda s: stable_unit("pick:" + s))[:limit_scenes]
        rtts = sorted((s for s in images if CITY_MARK not in s),
                      key=lambda s: stable_unit("pick:" + s))[:limit_scenes]
        keep = set(rtts) | {s for s in images if CITY_MARK in s and scene_of(s) in set(city_sc)}
        images = {s: v for s, v in images.items() if s in keep}
        log(f"--limit-scenes {limit_scenes}: kept {len(images)} images")

    store = data / "store" / "images"
    manifest, human = {}, {}
    orig_folders_by_scene = defaultdict(set)
    for stem, rec in tqdm(sorted(images.items()), desc="labels + images"):
        is_city = CITY_MARK in stem
        scene = scene_of(stem) if is_city else stem
        orig_folders_by_scene[scene].add(rec["orig_folder"])
        with Image.open(io.BytesIO(src.read_bytes(rec["key"]))) as im:
            W, H = im.size

        boxes, tag = None, None
        if is_city:
            lk = label_txt.get(stem, [None])[0] or city_label_by_scene.get(scene)
            if lk is not None:
                own = stem in label_txt
                lk_folder = Path(lk).parts[-3] if not src.zip else lk.split("/")[-3]
                tag = ("city_exact" if own else
                       "city_sibling_same_folder" if lk_folder == rec["orig_folder"]
                       else "city_sibling_other_folder (lost by old lookup)")
                boxes = parse_city_txt(src.read_bytes(lk).decode("utf-8", "ignore"), W, H, stats)
        else:
            xml = rtts_xml / f"{stem}.xml" if rtts_xml else None
            if xml is not None and xml.exists():
                boxes, (xw, xh) = parse_voc(xml, stats)
                if (xw, xh) != (W, H):
                    stats["rtts_xml_size_mismatch"] += 1
                tag = "rtts_xml"
            elif stem in label_txt:
                boxes = [[int(float(s.split()[0]))] + [float(v) for v in s.split()[1:5]]
                         for s in src.read_bytes(label_txt[stem][0]).decode().splitlines()
                         if len(s.split()) >= 5]
                tag = "rtts_txt_fallback"
        if boxes is None:
            stats["excluded_no_label_anywhere"] += 1
            continue
        stats[f"label_source:{tag}"] += 1
        boxes = dedupe(boxes, stats)

        src.copy_to(rec["key"], store / rec["name"])
        split = "dev" if stable_unit("scene:" + scene) < DEV_FRACTION else "train"
        manifest[stem] = {"name": rec["name"], "source": "city" if is_city else "rtts",
                          "scene": scene, "beta": beta_of(stem) if is_city else None,
                          "split": split, "orig_folder": rec["orig_folder"],
                          "width": W, "height": H, "n_boxes": len(boxes)}
        human[stem] = [b + [None] for b in boxes]
        write_labels(data / "human_labels" / f"{stem}.txt", boxes)

    # --- visibility ---------------------------------------------------------
    vis_info = {"computed": False}
    if visibility:
        rng = np.random.default_rng(0)
        scenes = defaultdict(dict)
        for stem, m in manifest.items():
            if m["source"] == "city" and m["beta"] in BETAS:
                scenes[m["scene"]][m["beta"]] = stem
        fits = []
        for scene, by_beta in tqdm(sorted(scenes.items()), desc="visibility"):
            if len(by_beta) != 3:
                stats["visibility_scene_incomplete"] += 1
                continue
            I = [lum(store / manifest[by_beta[b]]["name"]) for b in BETAS]
            if any(x is None for x in I) or len({x.shape for x in I}) != 1:
                stats["visibility_unreadable"] += 1
                continue
            t1, A, res = visibility_for_scene(*I, rng)
            fits.append(res)
            if res > max_fit_residual:
                stats["visibility_bad_fit"] += 1
                continue
            for b, stem in by_beta.items():
                power = round(b / BETAS[0])
                for box in human[stem]:
                    tb = box_transmittance(t1, box)
                    box[5] = None if tb is None else tb ** power
        vis_info = {"computed": True, "scenes_fitted": len(fits),
                    "fit_residual_median": float(np.median(fits)) if fits else None,
                    "max_fit_residual_accepted": max_fit_residual}

    # --- outputs ------------------------------------------------------------
    lists = {"train": [], "dev": []}
    for stem, m in sorted(manifest.items()):
        lists[m["split"]].append(m["name"])
    for k, v in lists.items():
        p = data / "lists" / f"{k}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(v) + "\n", encoding="utf-8")
    save_json(manifest, data / "manifest.json")
    save_json(human, data / "human_boxes.json")
    save_json(dev_coco_gt(manifest, human), data / "dev_gt.json")
    audit = make_audit(manifest, human, stats, orig_folders_by_scene, vis_info, src_path, rtts_xml)
    save_json(audit, data / "audit.json")
    log(f"train images {len(lists['train'])}, dev images {len(lists['dev'])}")
    log(f"wrote {data / 'audit.json'}")
    return audit


def dev_coco_gt(manifest, human):
    """COCO GT for the dev split in our class space (category_id = class + 1)."""
    imgs, anns, aid = [], [], 1
    for i, (stem, m) in enumerate(sorted((s, m) for s, m in manifest.items()
                                         if m["split"] == "dev")):
        W, H = m["width"], m["height"]
        imgs.append({"id": i + 1, "file_name": m["name"], "width": W, "height": H})
        for c, cx, cy, w, h, t in human[stem]:
            bw, bh = w * W, h * H
            anns.append({"id": aid, "image_id": i + 1, "category_id": int(c) + 1,
                         "bbox": [(cx - w / 2) * W, (cy - h / 2) * H, bw, bh],
                         "area": bw * bh, "iscrowd": 0, "transmittance": t})
            aid += 1
    return {"wase_format": "ours", "images": imgs, "annotations": anns,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]}


def make_audit(manifest, human, stats, folders_by_scene, vis_info, src_path, rtts_xml):
    by = Counter()
    cls = {"train": Counter(), "dev": Counter()}
    widths, vis = [], defaultdict(list)
    for stem, m in manifest.items():
        by[(m["source"], m["split"])] += 1
        if not human[stem]:
            by[("empty_label", m["split"])] += 1
        for c, cx, cy, w, h, t in human[stem]:
            cls[m["split"]][CLASSES[c]] += 1
            widths.append(w * m["width"] * 640 / max(m["width"], m["height"]))
            if t is not None:
                vis[m["beta"]].append(t)
    w = np.array(widths) if widths else np.zeros(1)
    vis_summary = {}
    for beta, ts in sorted(vis.items()):
        ts = np.array(ts)
        vis_summary[str(beta)] = {"boxes": int(ts.size), "median_t": float(np.median(ts)),
                                  **{f"pct_t_below_{th}": round(100 * float((ts < th).mean()), 2)
                                     for th in (0.05, 0.1, 0.2)}}
    return {
        "source_dataset": str(src_path), "rtts_xml": str(rtts_xml) if rtts_xml else None,
        "images_used": len(manifest),
        "images_by_source_and_split": {f"{a}/{b}": v for (a, b), v in sorted(by.items())},
        "boxes_by_split_and_class": {k: dict(v) for k, v in cls.items()},
        "boxes_total": int(sum(sum(v.values()) for v in cls.values())),
        "box_width_at_640": {"median": float(np.median(w)),
                             "pct_below_8px": round(100 * float((w < 8).mean()), 2),
                             "pct_below_12px": round(100 * float((w < 12).mean()), 2)},
        "scenes": len(folders_by_scene),
        "scenes_in_both_original_folders": sum(1 for v in folders_by_scene.values() if len(v) > 1),
        "visibility": {**vis_info, "by_beta": vis_summary},
        "counters": dict(sorted(stats.items())),
        "split_rule": f"scene-level, dev if md5('scene:'+scene) fraction < {DEV_FRACTION}",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=default_data_dir())
    ap.add_argument("--limit-scenes", type=int, default=None,
                    help="keep only N Cityscapes scenes and N RTTS images (smoke tests)")
    ap.add_argument("--no-visibility", action="store_true")
    a = ap.parse_args()
    build(a.data.resolve(), a.limit_scenes, visibility=not a.no_visibility)


if __name__ == "__main__":
    main()
