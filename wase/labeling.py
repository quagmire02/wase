"""
Pseudo-label generation for every condition, shared by train.py and diagnose.py
so the mechanism that is diagnosed is byte-for-byte the one that is trained.

Conditions (see PLAN.md for why each exists)
--------------------------------------------
naive            teacher on the raw frame
global-dcp       teacher on a full-frame DCP image (15x15 kernel)
crop-dqs-nogate  previous-round v2: crop each uncertain box, DCP, re-predict,
                 accept the first same-class box with higher confidence; DQS picks layout
crop-dqs         as above, but the accepted box must overlap the raw box (IoU >= 0.5)
crop-wbf         crop + gate, layouts fused with WBF instead of DQS
inplace-dqs      DCP the uncertain regions in place, one full-frame re-prediction,
                 gate IoU >= 0.5; DQS picks layout
inplace-wbf      in place + gate, WBF fusion
rawflip-wbf      raw frame + horizontally flipped raw frame, WBF (ensembling control:
                 two views, no enhancement)

The labels written to disk are always for the RAW image.
"""

from dataclasses import dataclass

import numpy as np

from .common import (DCP_KERNEL_FULL, DCP_KERNEL_ROI, dehaze, dqs, iou_cxcywh,
                     nms_classwise, predict, weighted_boxes_fusion)

NMS_IOU = 0.5
GATE_IOU = 0.5
WBF_IOU = 0.55
REPREDICT_CONF = 0.10
MIN_ROI_PX = 10


@dataclass(frozen=True)
class Spec:
    view: str              # raw | global | crop | inplace | flip
    gate: bool = False
    combine: str = "none"  # none | dqs | wbf


PSEUDO_CONDITIONS = {
    "naive": Spec("raw"),
    "global-dcp": Spec("global"),
    "crop-dqs-nogate": Spec("crop", gate=False, combine="dqs"),
    "crop-dqs": Spec("crop", gate=True, combine="dqs"),
    "crop-wbf": Spec("crop", gate=True, combine="wbf"),
    "inplace-dqs": Spec("inplace", gate=True, combine="dqs"),
    "inplace-wbf": Spec("inplace", gate=True, combine="wbf"),
    "rawflip-wbf": Spec("flip", combine="wbf"),
}


def _roi_px(box, W, H, pad=0.0):
    _, cx, cy, w, h = box[:5]
    pw, ph = w * pad / 2, h * pad / 2
    x1, y1 = max(0, int((cx - w / 2 - pw) * W)), max(0, int((cy - h / 2 - ph) * H))
    x2, y2 = min(W, int((cx + w / 2 + pw) * W)), min(H, int((cy + h / 2 + ph) * H))
    return x1, y1, x2, y2


def refine_crop(model, img, raw, cmap, band, gate, imgsz, use_dcp=True, pad=0.0):
    """Crop-and-re-predict refinement. Returns (boxes, per-uncertain-box records)."""
    H, W = img.shape[:2]
    lo, hi = band
    out, recs = [], []
    for b in raw:
        conf = b[5]
        if conf > hi:
            out.append(b)
            continue
        if conf < lo:
            continue
        x1, y1, x2, y2 = _roi_px(b, W, H, pad)
        if x2 - x1 < MIN_ROI_PX or y2 - y1 < MIN_ROI_PX:
            out.append(b)
            continue
        roi = img[y1:y2, x1:x2]
        roi = dehaze(roi, DCP_KERNEL_ROI) if use_dcp else roi
        cands = predict(model, roi, REPREDICT_CONF, cmap, imgsz=imgsz)
        rw, rh = x2 - x1, y2 - y1
        mapped = [[c[0], (x1 + c[1] * rw) / W, (y1 + c[2] * rh) / H,
                   c[3] * rw / W, c[4] * rh / H, c[5]] for c in cands if c[0] == b[0]]
        mapped.sort(key=lambda m: -m[5])
        best_any = mapped[0] if mapped else None
        chosen = None
        for m in mapped:
            if m[5] <= conf:
                break
            if not gate or iou_cxcywh(m[1:5], b[1:5]) >= GATE_IOU:
                chosen = m
                break
        recs.append({"conf_raw": conf, "conf_best": best_any[5] if best_any else None,
                     "accepted": chosen is not None,
                     "area_ratio_best": (best_any[3] * best_any[4]) / max(b[3] * b[4], 1e-12) if best_any else None,
                     "best_box": list(best_any[:5]) if best_any else None,
                     "area_ratio": (chosen[3] * chosen[4]) / max(b[3] * b[4], 1e-12) if chosen else None})
        out.append(chosen if chosen else b)
    return out, recs


def refine_inplace(model, img, raw, cmap, band, gate, imgsz, use_dcp=True):
    """
    DCP the uncertain regions in place and re-predict the full frame once.
    In place is always spatially gated: without the IoU match there is no way
    to say which enhanced box corresponds to which raw box. `gate` is accepted
    only so both refiners share a signature.
    """
    H, W = img.shape[:2]
    lo, hi = band
    comp = img.copy()
    touched = False
    for b in raw:
        if lo <= b[5] <= hi:
            x1, y1, x2, y2 = _roi_px(b, W, H)
            if x2 - x1 >= MIN_ROI_PX and y2 - y1 >= MIN_ROI_PX:
                comp[y1:y2, x1:x2] = dehaze(img[y1:y2, x1:x2], DCP_KERNEL_ROI) if use_dcp \
                    else img[y1:y2, x1:x2]
                touched = True
    if not touched:
        return [b for b in raw if b[5] >= lo], []
    enhanced = predict(model, comp, REPREDICT_CONF, cmap, imgsz=imgsz)
    out, recs = [], []
    for b in raw:
        if b[5] > hi:
            out.append(b)
            continue
        if b[5] < lo:
            continue
        overl = [e for e in enhanced if e[0] == b[0] and iou_cxcywh(e[1:5], b[1:5]) >= GATE_IOU]
        best_any = max(overl, key=lambda e: e[5]) if overl else None
        chosen = best_any if best_any is not None and best_any[5] > b[5] else None
        recs.append({"conf_raw": b[5], "conf_best": best_any[5] if best_any else None,
                     "accepted": chosen is not None,
                     "area_ratio_best": (best_any[3] * best_any[4]) / max(b[3] * b[4], 1e-12) if best_any else None,
                     "best_box": list(best_any[:5]) if best_any else None,
                     "area_ratio": (chosen[3] * chosen[4]) / max(b[3] * b[4], 1e-12) if chosen else None})
        out.append(chosen if chosen else b)
    return out, recs


def flip_view(model, img, conf, cmap, imgsz):
    boxes = predict(model, np.ascontiguousarray(img[:, ::-1]), conf, cmap, imgsz=imgsz)
    return nms_classwise([[b[0], 1.0 - b[1], b[2], b[3], b[4], b[5]] for b in boxes], NMS_IOU)


def label_image(model, img, cmap, spec, conf, band, imgsz):
    """
    Returns (final_boxes, info). info carries what the per-generation stats need:
    raw/second-view box counts, whether DQS chose the second view, and the
    per-uncertain-box records used for the unconditional-confidence analysis.
    """
    if spec.view == "global":
        img_in = dehaze(img, DCP_KERNEL_FULL)
        boxes = nms_classwise(predict(model, img_in, conf, cmap, imgsz=imgsz), NMS_IOU)
        return boxes, {"n_raw": len(boxes)}
    raw = nms_classwise(predict(model, img, conf, cmap, imgsz=imgsz), NMS_IOU)
    info = {"n_raw": len(raw)}
    if spec.view == "raw" or not raw:
        return raw, info

    if spec.view == "crop":
        second, recs = refine_crop(model, img, raw, cmap, band, spec.gate, imgsz)
    elif spec.view == "inplace":
        second, recs = refine_inplace(model, img, raw, cmap, band, spec.gate, imgsz)
    elif spec.view == "flip":
        second, recs = flip_view(model, img, conf, cmap, imgsz), []
    else:
        raise ValueError(spec.view)
    if spec.view != "flip":
        second = nms_classwise(second, NMS_IOU)
    info.update(n_second=len(second), records=recs,
                dqs_prefers_second=dqs(second) > dqs(raw))

    if spec.combine == "dqs":
        final = second if info["dqs_prefers_second"] else raw
    elif spec.combine == "wbf":
        final = nms_classwise(weighted_boxes_fusion([raw, second], WBF_IOU), NMS_IOU)
    else:
        final = second
    return final, info
