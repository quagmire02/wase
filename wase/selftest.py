"""
Checks that need no dataset and no GPU. Run before trusting any number:

  python -m wase.selftest

1. Our AP engine equals pycocotools on random GT/detections (crowd regions,
   all size ranges, ties excluded by continuous scores).
2. Bootstrap weights reproduce explicit image duplication exactly.
3. The DQS identity: a layout built only from higher-confidence replacements
   always scores DQS >= the raw layout (the reason DQS cannot judge quality).
4. The three-beta transmittance inversion recovers a known synthetic fog.
5. Class maps: COCO teacher -> ours, 5-class model -> identity.
6. Verification rule: agreement admits and rejects boxes, never moves them.
7. Oracle repairs: each one fixes exactly its own error type.
8. Seed-resampling bootstrap: equal to the fixed-seed one for a single seed,
   wider when seeds disagree.
9. Crop refinement can only shrink a box (zero-padding crops; PLAN.md 2, flaw 3).
"""

import contextlib
import io
import sys

import numpy as np


def _random_set(rng, n_img=60, n_cls=5):
    gt, dets = {}, {}
    for i in range(1, n_img + 1):
        m = rng.integers(0, 12)
        g = np.column_stack([rng.uniform(0, 900, m), rng.uniform(0, 500, m),
                             rng.uniform(4, 300, m), rng.uniform(4, 250, m)])
        crowd = rng.random(m) < 0.05
        gt[i] = {"boxes": g, "cls": rng.integers(0, n_cls, m),
                 "area": g[:, 2] * g[:, 3] * rng.uniform(0.6, 1.0, m), "crowd": crowd}
        n = rng.integers(0, 40)
        jitter = []
        for k in range(n):
            if m and rng.random() < 0.6:
                j = rng.integers(0, m)
                b = g[j] + rng.normal(0, 12, 4)
                c = gt[i]["cls"][j] if rng.random() < 0.85 else rng.integers(0, n_cls)
            else:
                b = np.array([rng.uniform(0, 900), rng.uniform(0, 500),
                              rng.uniform(4, 300), rng.uniform(4, 250)])
                c = rng.integers(0, n_cls)
            b[2:] = np.abs(b[2:]) + 1
            jitter.append((b, c))
        dets[i] = {"boxes": np.array([b for b, _ in jitter]).reshape(-1, 4),
                   "cls": np.array([c for _, c in jitter], dtype=int),
                   "score": rng.random(n)}
    return gt, dets


def _pycoco(gt, dets, n_cls):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images = [{"id": i, "width": 1000, "height": 600} for i in gt]
    anns, aid = [], 1
    for i, g in gt.items():
        for b, c, a, cr in zip(g["boxes"], g["cls"], g["area"], g["crowd"]):
            anns.append({"id": aid, "image_id": i, "category_id": int(c) + 1,
                         "bbox": [float(v) for v in b], "area": float(a), "iscrowd": int(cr)})
            aid += 1
    cats = [{"id": c + 1, "name": str(c)} for c in range(n_cls)]
    res = [{"image_id": i, "category_id": int(c) + 1, "bbox": [float(v) for v in b], "score": float(s)}
           for i, d in dets.items() for b, c, s in zip(d["boxes"], d["cls"], d["score"])]
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = {"images": images, "annotations": anns, "categories": cats}
        cg.createIndex()
        cd = cg.loadRes(res)
        ev = COCOeval(cg, cd, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {"mAP50_95": s[0], "mAP50": s[1], "mAP75": s[2],
            "mAP_small": s[3], "mAP_medium": s[4], "mAP_large": s[5]}


def check_ap_engine():
    from .metrics import Scorer
    worst = 0.0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        gt, dets = _random_set(rng)
        ours = Scorer(gt, dets, 5).summary()
        ref = _pycoco(gt, dets, 5)
        for k, v in ref.items():
            if v < 0:          # pycocotools reports -1 for "no GT in this range"
                continue
            worst = max(worst, abs(ours[k] - v))
    assert worst < 1e-9, f"AP engine differs from pycocotools by {worst}"
    return f"AP engine == pycocotools (max |diff| {worst:.1e} over 4 random sets, 6 metrics)"


def check_bootstrap_weights():
    from .metrics import Scorer
    rng = np.random.default_rng(7)
    gt, dets = _random_set(rng, n_img=30)
    s = Scorer(gt, dets, 5, areas=("all",))
    w = rng.integers(0, 3, s.n_img)
    via_weights = s.map50_resampled(w)
    gt2, dets2, nid = {}, {}, 1
    for k, iid in enumerate(s.image_ids):
        for _ in range(w[k]):
            gt2[nid], dets2[nid] = gt[iid], dets[iid]
            nid += 1
    explicit = Scorer(gt2, dets2, 5, areas=("all",)).summary()["mAP50"]
    assert abs(via_weights - explicit) < 1e-6, (via_weights, explicit)
    return f"bootstrap re-weighting == explicit duplication ({via_weights:.6f})"


def check_dqs_identity():
    from .common import dqs
    rng = np.random.default_rng(3)
    for _ in range(2000):
        n = rng.integers(1, 15)
        raw = [[0, 0, 0, 0, 0, float(c)] for c in rng.uniform(0.15, 0.9, n)]
        refined = [list(b) for b in raw]
        for b in refined:
            if 0.10 <= b[5] <= 0.65 and rng.random() < 0.4:
                b[5] = float(rng.uniform(b[5], 1.0))   # accepted only if higher
        assert dqs(refined) >= dqs(raw) - 1e-12
    return "DQS(refined) >= DQS(raw) whenever replacements are confidence-gated (2000 trials)"


def check_transmittance():
    from .prep import visibility_for_scene
    rng = np.random.default_rng(0)
    H, W, A = 120, 200, 238.0
    J = rng.uniform(0, 200, (H, W))
    depth = np.tile(np.linspace(5, 400, W), (H, 1))
    t = np.exp(-0.005 * depth)
    imgs = [J * t ** k + A * (1 - t ** k) for k in (1, 2, 4)]
    t_hat, A_hat, res = visibility_for_scene(*imgs, rng)
    ok = np.isfinite(t_hat)
    err = np.abs(t_hat[ok] - t[ok]).max()
    assert abs(A_hat - A) <= 0.5 and err < 1e-6, (A_hat, err)
    return f"transmittance inversion: airlight {A_hat} (true {A}), max |t error| {err:.1e}"


def check_class_maps():
    from .common import COCO_TO_OURS, IDENTITY, class_map_for

    class M:
        def __init__(self, n):
            self.names = {i: str(i) for i in range(n)}
    assert class_map_for(M(80)) is COCO_TO_OURS
    assert class_map_for(M(5)) is IDENTITY
    assert 4 in COCO_TO_OURS.values() and COCO_TO_OURS[5] == 4
    return "class maps resolved from the model's own class count"


def check_verify_rule():
    from .labeling import verify_boxes
    A = [2, 0.2, 0.2, 0.1, 0.1]      # object seen in both views
    B = [2, 0.6, 0.6, 0.1, 0.1]      # object seen only in the raw view
    cands = [A + [0.12], B + [0.12], [0, 0.4, 0.4, 0.1, 0.1, 0.30], [0, 0.8, 0.8, 0.1, 0.1, 0.40],
             [1, 0.5, 0.2, 0.1, 0.1, 0.90], [1, 0.9, 0.2, 0.1, 0.1, 0.05]]
    second = [A + [0.5], [0, 0.4, 0.4, 0.1, 0.1, 0.2], [2, 0.8, 0.8, 0.1, 0.1, 0.9]]  # last: wrong class
    conf, band = 0.15, (0.10, 0.65)
    confs = lambda bs: sorted(b[5] for b in bs)
    both, n = verify_boxes(cands, second, conf, band, "both")
    assert confs(both) == [0.12, 0.30, 0.90] and n == {"n_naive": 3, "n_admitted": 1, "n_rejected": 1}, (both, n)
    assert confs(verify_boxes(cands, second, conf, band, "admit")[0]) == [0.12, 0.30, 0.40, 0.90]
    assert confs(verify_boxes(cands, second, conf, band, "reject")[0]) == [0.30, 0.90]
    naive = [b for b in cands if b[5] >= conf]
    assert verify_boxes(naive, [list(b) for b in naive], conf, band, "both")[0] == naive
    assert all(b in cands for b in both), "verification must not change geometry"
    return "verification: agreement admits 1, rejects 1, keeps raw geometry; full agreement == naive"


def check_oracles():
    from .common import iou_cxcywh
    from .labeling import apply_oracle
    human = [[2, 0.30, 0.30, 0.20, 0.20, None], [0, 0.70, 0.70, 0.10, 0.20, None], [4, 0.50, 0.10, 0.10, 0.10, None]]
    pseudo = [[2, 0.31, 0.30, 0.16, 0.18, 0.8],      # car, found but too small
              [0, 0.20, 0.80, 0.05, 0.10, 0.3],      # person that is not there
              [2, 0.70, 0.70, 0.10, 0.20, 0.5]]      # the person, called a car: false + missed
    nofp, c = apply_oracle("nofp", pseudo, human)
    assert c == {"oracle_matched": 1, "oracle_false": 2, "oracle_missed": 2}, c
    assert nofp == [pseudo[0]]
    addfn, _ = apply_oracle("addfn", pseudo, human)
    assert addfn[:3] == pseudo and [b[:5] for b in addfn[3:]] == [human[1][:5], human[2][:5]]
    geom, _ = apply_oracle("geom", pseudo, human)
    assert geom[0] == [2, 0.30, 0.30, 0.20, 0.20, 0.8] and geom[1:] == pseudo[1:]
    assert iou_cxcywh(pseudo[0][1:5], human[0][1:5]) < 1 and apply_oracle("nofp", [], human)[0] == []
    assert pseudo[0] == [2, 0.31, 0.30, 0.16, 0.18, 0.8], "oracle must not modify its input"
    return "oracles: nofp drops 2 false, addfn adds 2 missed, geom snaps 1 matched box; nothing else changes"


def check_seed_bootstrap():
    from .metrics import Scorer, bootstrap
    rng = np.random.default_rng(11)
    gt, dets = _random_set(rng, n_img=40)
    worse = {i: {"boxes": d["boxes"] + np.array([150.0, 0, 0, 0]), "cls": d["cls"], "score": d["score"]}
             for i, d in dets.items()}
    a, b = Scorer(gt, dets, 5, areas=("all",)), Scorer(gt, worse, 5, areas=("all",))
    one_fixed = bootstrap({"m": [a]}, 200, resample_seeds=False)["m"]
    one_res = bootstrap({"m": [a]}, 200, resample_seeds=True)["m"]
    assert np.array_equal(one_fixed, one_res), "one seed: both bootstraps must agree"
    fixed = bootstrap({"m": [a, b]}, 400, resample_seeds=False)["m"]
    res = bootstrap({"m": [a, b]}, 400, resample_seeds=True)["m"]
    assert res.std() > fixed.std(), (res.std(), fixed.std())
    return (f"seed-resampling bootstrap: identical for 1 seed; sd {fixed.std():.4f} -> {res.std():.4f} "
            f"when two seeds disagree")


def check_crop_can_only_shrink():
    from . import labeling
    rng = np.random.default_rng(5)
    img = np.zeros((480, 640, 3), np.uint8)
    real = labeling.predict
    # a stub detector: random same-class boxes anywhere in whatever it is shown
    labeling.predict = lambda model, im, conf, cmap, imgsz=640: [
        [2, *rng.uniform(0.2, 0.8, 2), *rng.uniform(0.05, 0.4, 2), float(rng.uniform(0.7, 1.0))] for _ in range(6)]
    try:
        worst = 0.0
        for _ in range(300):
            raw = [[2, float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.3, 0.7)),
                    float(rng.uniform(0.1, 0.3)), float(rng.uniform(0.1, 0.3)), 0.3]]
            _, recs = labeling.refine_crop(None, img, raw, None, (0.10, 0.65), False, 640, use_dcp=False)
            worst = max(worst, max(r["area_ratio_best"] for r in recs))
    finally:
        labeling.predict = real
    assert worst <= 1.0 + 1e-9, worst
    return f"crop refinement: re-detected box never larger than the raw box (max area ratio {worst:.3f} in 300 trials)"


def check_error_breakdown():
    from .diagnose import classify_errors
    gts = [(2, 0, 0, 100, 100), (0, 200, 0, 240, 100), (4, 400, 0, 500, 100), (1, 600, 0, 640, 40)]
    dets = [(2, 2, 2, 100, 100, 0.9),        # correct
            (2, 0, 0, 98, 98, 0.8),          # duplicate of the car
            (2, 200, 0, 240, 100, 0.7),      # the person, called a car
            (4, 400, 0, 440, 100, 0.6),      # the bus, 40% covered
            (0, 300, 300, 320, 340, 0.5)]    # nothing there
    labels, found, why = classify_errors(gts, dets)
    assert labels == ["correct", "duplicate", "wrong_class", "mislocated", "background"], labels
    assert found == [True, False, False, False] and why == [None, "wrong_class", "mislocated", "unseen"], why
    return "error breakdown: one label per detection, one fate per ground-truth box"


def main():
    checks = [check_error_breakdown, check_ap_engine, check_bootstrap_weights, check_dqs_identity,
              check_transmittance, check_class_maps, check_verify_rule, check_oracles,
              check_seed_bootstrap, check_crop_can_only_shrink]
    failed = 0
    for c in checks:
        try:
            print(f"PASS  {c()}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {c.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
