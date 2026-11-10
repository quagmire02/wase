"""
COCO-faithful AP engine (bbox, 101-point interpolation, maxDets=100, COCO area
ranges and crowd/area ignore rules) plus a fast image-level bootstrap.

Why not call pycocotools for everything: a paired bootstrap needs thousands of
re-scorings. This module reproduces pycocotools' evaluateImg/accumulate exactly
(checked in selftest.py and on every evaluate.py run), precomputes per-image
matches once, and re-scores a resample by repeating detections.

GT and detections use our 5-class ids (0..4) and pixel xywh boxes.
"""

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
AREAS = {"all": (0.0, 1e10), "small": (0.0, 32.0 ** 2),
         "medium": (32.0 ** 2, 96.0 ** 2), "large": (96.0 ** 2, 1e10)}
MAX_DETS = 100


def _iou_matrix(d, g, crowd):
    """IoU of dets d (N,4 xywh) vs gts g (M,4 xywh); crowd gts use det area as union."""
    if len(d) == 0 or len(g) == 0:
        return np.zeros((len(d), len(g)))
    dx1, dy1 = d[:, 0:1], d[:, 1:2]
    dx2, dy2 = dx1 + d[:, 2:3], dy1 + d[:, 3:4]
    gx1, gy1 = g[:, 0], g[:, 1]
    gx2, gy2 = gx1 + g[:, 2], gy1 + g[:, 3]
    iw = np.clip(np.minimum(dx2, gx2) - np.maximum(dx1, gx1), 0, None)
    ih = np.clip(np.minimum(dy2, gy2) - np.maximum(dy1, gy1), 0, None)
    inter = iw * ih
    da = (d[:, 2] * d[:, 3])[:, None]
    ga = (g[:, 2] * g[:, 3])[None, :]
    union = np.where(crowd[None, :], da, da + ga - inter)
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_image(gt_boxes, gt_area, gt_crowd, dt_boxes, dt_scores, area_rng, iou_thrs):
    """
    pycocotools COCOeval.evaluateImg for one image and one category.
    Returns (scores[D], matched[T,D], dt_ignore[T,D], n_gt_not_ignored).
    """
    lo, hi = area_rng
    g_ig = gt_crowd | (gt_area < lo) | (gt_area > hi)
    gi = np.argsort(g_ig, kind="mergesort")
    gt_boxes, gt_crowd, g_ig = gt_boxes[gi], gt_crowd[gi], g_ig[gi]
    di = np.argsort(-dt_scores, kind="mergesort")[:MAX_DETS]
    dt_boxes, dt_scores = dt_boxes[di], dt_scores[di]
    T, D, G = len(iou_thrs), len(dt_scores), len(gt_boxes)
    gtm = np.zeros((T, G), bool)
    dtm = np.zeros((T, D), bool)
    dt_ig = np.zeros((T, D), bool)
    if G and D:
        ious = _iou_matrix(dt_boxes, gt_boxes, gt_crowd)
        thr = np.minimum(np.asarray(iou_thrs), 1 - 1e-10)[:, None]
        rows = np.arange(T)
        for d in range(D):
            # pycocotools' inner loop, vectorised over thresholds and GTs: take the
            # highest-IoU eligible non-ignored GT; only if none exists, the highest
            # ignored one. Ties go to the LAST such GT, as in the original loop.
            iou_d = ious[d][None, :]
            cand = (~gtm | gt_crowd[None, :]) & (iou_d >= thr)
            non_ig = cand & ~g_ig[None, :]
            use = np.where(non_ig.any(1, keepdims=True), non_ig, cand & g_ig[None, :])
            has = use.any(1)
            if not has.any():
                continue
            vals = np.where(use, iou_d, -1.0)
            m = G - 1 - np.argmax(vals[:, ::-1], axis=1)
            t_hit = rows[has]
            m_hit = m[has]
            dt_ig[t_hit, d] = g_ig[m_hit]
            dtm[t_hit, d] = True
            gtm[t_hit, m_hit] = True
    dt_area = dt_boxes[:, 2] * dt_boxes[:, 3] if D else np.zeros(0)
    out_rng = (dt_area < lo) | (dt_area > hi)
    dt_ig |= (~dtm) & out_rng[None, :]
    return dt_scores, dtm, dt_ig, int((~g_ig).sum())


def ap_101(tp_sorted, npig):
    """AP from detections already sorted by score with ignored ones removed."""
    if npig == 0:
        return None
    if tp_sorted.size == 0:
        return 0.0
    ctp = np.cumsum(tp_sorted, dtype=float)
    cfp = np.cumsum(~tp_sorted, dtype=float)
    rc = ctp / npig
    pr = ctp / (ctp + cfp + np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    idx = np.searchsorted(rc, REC_THRS, side="left")
    q = np.where(idx < pr.size, pr[np.minimum(idx, pr.size - 1)], 0.0)
    return float(q.mean())


class Scorer:
    """
    Precomputed matches for one model on one GT set.

    gt:   {image_id: {"boxes": (M,4) xywh, "cls": (M,), "area": (M,), "crowd": (M,) bool}}
    dets: {image_id: {"boxes": (N,4) xywh, "cls": (N,), "score": (N,)}}
    """

    def __init__(self, gt, dets, n_classes, areas=("all", "small", "medium", "large")):
        self.image_ids = sorted(gt)
        self.n_img = len(self.image_ids)
        self.n_classes = n_classes
        self.areas = areas
        # tables[(area, cls)] = (img_index[K], score[K], matched[T,K], ignore[T,K], npig_per_img[n_img])
        self.tables = {}
        empty4 = np.zeros((0, 4))
        for a in areas:
            for c in range(n_classes):
                rows_img, rows_s, rows_m, rows_i = [], [], [], []
                npig = np.zeros(self.n_img, dtype=np.int64)
                for k, iid in enumerate(self.image_ids):
                    g = gt[iid]
                    gs = g["cls"] == c
                    d = dets.get(iid)
                    if d is not None and len(d["cls"]):
                        ds = d["cls"] == c
                        db, dsc = d["boxes"][ds], d["score"][ds]
                    else:
                        db, dsc = empty4, np.zeros(0)
                    if not gs.any() and dsc.size == 0:
                        continue
                    s, m, ig, n = match_image(g["boxes"][gs], g["area"][gs], g["crowd"][gs],
                                              db, dsc, AREAS[a], IOU_THRS)
                    npig[k] = n
                    rows_img.append(np.full(s.size, k))
                    rows_s.append(s)
                    rows_m.append(m)
                    rows_i.append(ig)
                if rows_s:
                    s = np.concatenate(rows_s)
                    order = np.argsort(-s, kind="mergesort")
                    tab = (np.concatenate(rows_img)[order], s[order],
                           np.concatenate(rows_m, axis=1)[:, order],
                           np.concatenate(rows_i, axis=1)[:, order], npig)
                else:
                    tab = (np.zeros(0, int), np.zeros(0), np.zeros((len(IOU_THRS), 0), bool),
                           np.zeros((len(IOU_THRS), 0), bool), npig)
                self.tables[(a, c)] = tab

    # -- full-set metrics ------------------------------------------------------
    def class_ap(self, area="all", t_index=0, weights=None):
        """Per-class AP at one IoU threshold; weights = per-image multiplicity."""
        out = []
        for c in range(self.n_classes):
            img, _, m, ig, npig = self.tables[(area, c)]
            if weights is None:
                keep = ~ig[t_index]
                out.append(ap_101(m[t_index][keep], int(npig.sum())))
            else:
                w = weights[img]
                keep = (~ig[t_index]) & (w > 0)
                idx = np.flatnonzero(keep)
                rep = np.repeat(idx, w[idx])
                out.append(ap_101(m[t_index][rep], int((npig * weights).sum())))
        return out

    def summary(self):
        res = {}
        per_t = {a: [self.class_ap(a, t) for t in range(len(IOU_THRS))] for a in self.areas}

        def mean_valid(vals):
            v = [x for x in vals if x is not None]
            return float(np.mean(v)) if v else None

        def mean_over_t(a):
            v = [x for t in per_t[a] for x in t if x is not None]
            return float(np.mean(v)) if v else None

        res["mAP50_95"] = mean_over_t("all")
        res["mAP50"] = mean_valid(per_t["all"][0])
        res["mAP75"] = mean_valid(per_t["all"][5])
        for a in self.areas:
            if a != "all":
                res[f"mAP_{a}"] = mean_over_t(a)          # COCO APs/APm/APl (IoU .50:.95)
                res[f"mAP50_{a}"] = mean_valid(per_t[a][0])
        counts = [int(self.tables[("all", c)][4].sum()) for c in range(self.n_classes)]
        ap50 = per_t["all"][0]
        res["per_class_AP50"] = ap50
        res["per_class_gt"] = counts
        tot = sum(n for n, a in zip(counts, ap50) if a is not None)
        res["instance_weighted_AP50"] = (float(sum(n * a for n, a in zip(counts, ap50) if a is not None) / tot)
                                         if tot else None)
        return res

    def map50_resampled(self, weights, area="all"):
        v = [x for x in self.class_ap(area, 0, weights) if x is not None]
        return float(np.mean(v)) if v else np.nan


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def draw_weights(n_img, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)


def bootstrap(groups, n_boot=2000, seed=0, area="all", resample_seeds=True):
    """
    groups: {label: [Scorer per seed]} all built on the same GT.
    Each resample draws images once; every seed of every label is re-scored on
    the same image draw. With resample_seeds (default) the training seeds of
    each label are ALSO drawn with replacement, independently per label, so the
    interval reflects seed-to-seed variation as well as the choice of test
    images (a two-level bootstrap). With one seed this changes nothing; with
    three it is a coarse but honest estimate. resample_seeds=False reproduces
    the previous round: seeds treated as fixed and simply averaged.
    Returns {label: np.array(n_boot)} of seed-averaged mAP50.
    """
    n_img = next(iter(groups.values()))[0].n_img
    out = {k: np.empty(n_boot) for k in groups}
    seed_rng = np.random.default_rng([seed, 1])
    for b, w in enumerate(draw_weights(n_img, n_boot, seed)):
        for k, scorers in groups.items():
            vals = np.array([s.map50_resampled(w, area) for s in scorers])
            if resample_seeds and len(vals) > 1:
                vals = vals[seed_rng.integers(0, len(vals), len(vals))]
            out[k][b] = np.nanmean(vals)
    return out


def delta_stats(a, b, margin=None, alpha=0.05):
    """b - a: mean, 95% CI, two-sided bootstrap p, optional TOST equivalence."""
    d = b - a
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    out = {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
           "p_two_sided": float(max(p, 1.0 / d.size)), "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    if margin is not None:
        l90, h90 = np.percentile(d, [5, 95])
        out["tost"] = {"margin": margin, "ci90": [float(l90), float(h90)],
                       "equivalent": bool(l90 > -margin and h90 < margin)}
    return out


# ---------------------------------------------------------------------------
# Label agreement (pseudo-labels vs human labels), greedy one-to-one at IoU>=thr
# ---------------------------------------------------------------------------
def match_pairs(gt, pred, thr=0.5):
    """
    Greedy one-to-one, class-aware matching in descending prediction score.
    gt: list of (cls, x1, y1, x2, y2); pred: list of (cls, x1, y1, x2, y2, score).
    Returns pred_to_gt: for every prediction the index of its GT box, or -1.
    """
    from .common import iou_xyxy
    taken = [False] * len(gt)
    pred_to_gt = [-1] * len(pred)
    for j in sorted(range(len(pred)), key=lambda k: -pred[k][5]):
        best, bi = thr, -1
        for i, g in enumerate(gt):
            if taken[i] or g[0] != pred[j][0]:
                continue
            v = iou_xyxy(pred[j][1:5], g[1:5])
            if v >= best:
                best, bi = v, i
        if bi >= 0:
            taken[bi] = True
            pred_to_gt[j] = bi
    return pred_to_gt


def greedy_match(gt, pred, thr=0.5):
    """Returns (gt_matched[bool], pred_matched[bool]); see match_pairs."""
    p2g = match_pairs(gt, pred, thr)
    gm = [False] * len(gt)
    for i in p2g:
        if i >= 0:
            gm[i] = True
    return gm, [i >= 0 for i in p2g]


def expected_calibration_error(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, cf = correct[m].mean(), conf[m].mean()
            ece += m.mean() * abs(acc - cf)
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                         "accuracy": float(acc), "confidence": float(cf)})
    return float(ece), bins
