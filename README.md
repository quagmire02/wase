# WASE resubmission

This is clean code for the fog pseudo-labelling study. The research plan (why the
original WASE lost to plain pseudo-labelling, what we test now, and in what order)
is in **[PLAN.md](PLAN.md)**.

## Layout

```
wase/
  common.py     paths, classes, class maps, DCP, NMS, WBF, DQS, label I/O, seeding
  prep.py       one-time data build: fixed label lookup, scene split, visibility, audit.json
  labeling.py   pseudo-label generation for every condition (shared by train + diagnose)
  train.py      any condition, any seed, 3-generation self-training or supervised
  adabn.py      label-free baseline: BatchNorm statistics adapted to fog, nothing else
  diagnose.py   no-training mechanism studies (scale, selection, calibration, agreement)
  metrics.py    COCO-faithful AP engine + image bootstrap + equivalence tests
  evaluate.py   pycocotools evaluation -> JSON; --report builds tables
  selftest.py   checks that need no data (run first)
  inspect_labels.py  draws labels on random images for a by-eye check (results/inspect/)
scripts/
  smoke.ps1             10-minute end-to-end test on a tiny subset
  run_experiments.ps1   every run in PLAN.md, in order
ANALYSIS_PLAN.md        pre-registration; commit before held-out evaluation
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
# RTX 20xx/30xx/40xx: CUDA 12.1 wheels. RTX 50xx (Blackwell) needs the cu128 index and torch >= 2.7.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m wase.selftest
```

### Expected data locations

The scripts look for these next to this folder; override them with environment variables.

| What | Default location | Override |
|---|---|---|
| Ultimate_Fog_Dataset (folder **or** the original `.zip`) | `../Ultimate_Fog_Dataset[.zip]` | `WASE_SOURCE_DATA` |
| RTTS VOC XML | `../RTTS/Annotations` | `WASE_RTTS_XML` |
| COCO YOLOv8n | `../yolov8n.pt` (otherwise downloaded) | `--teacher` |
| ACDC | anywhere | `--gt` / `--img-root` |
| Derived data, runs and results | `data/`, `runs/`, `results/` | `WASE_DATA`, `WASE_RUNS`, `WASE_RESULTS` |

## Typical use

```powershell
python -m wase.prep                                  # ~19 GB copied into data/store; writes data/audit.json
python -m wase.train --condition naive --seed 0
python -m wase.train --condition supervised --seed 0
python -m wase.evaluate --set acdc_fog_val --gt <ACDC>\gt_detection\fog\val\instancesonly_fog_val_gt_detection.json `
    --img-root <ACDC>\rgb_anon\fog\val --models coco=..\yolov8n.pt "runs\naive\seed*\weights\final.pt" `
    "runs\supervised\seed*\weights\final.pt" --compare supervised:naive --margin 0.01
python -m wase.evaluate --report                     # results/tables.md and tables.tex
```

Run `scripts\smoke.ps1` once on any new machine before starting real runs.

## Conditions

| Family | `--condition` | What it does |
|---|---|---|
| Baselines | `naive`, `supervised`, `global-dcp` | teacher labels / human labels / teacher on a de-fogged frame |
| Refinement (previous round) | `crop-dqs-nogate`, `crop-dqs`, `crop-wbf`, `inplace-dqs`, `inplace-wbf`, `rawflip-wbf` | re-detect uncertain boxes in a second view and **move** them |
| Verification (v2) | `verify-flip`, `verify-dcp`, `verify-scale` | use the second view to **keep or drop** uncertain boxes; geometry untouched. `--verify-mode admit\|reject` for the ablation |
| Oracles (v2) | `oracle-nofp`, `oracle-addfn`, `oracle-geom` | naive labels with one error type repaired from human labels. Upper bounds, not methods |

Useful options: `--teacher yolov8m.pt --label-imgsz 1280 --student yolov8n.pt` (strong teacher at
label time, small deployed model), `--label-filter`, `--source`, `--fraction`, `--lr-scale`, `--imgsz`.
`python -m wase.adabn` writes `runs/adabn/seed0/weights/final.pt`, which `evaluate` picks up like any run.

Why the v2 families exist, and what each outcome would mean, is in [PLAN.md](PLAN.md) section 10.

## Guarantees the code enforces

- **Same images, same schedule, same checkpoint rule for every condition.** `last.pt` is always used, and there is no early stopping.
- **No cross-generation weight averaging.** The class map comes from the running model's class count.
- **Each run owns its label folder**, and caches are wiped every generation.
- **Training aborts** if a generation has zero boxes or more than 90% empty labels.
- **Headline numbers are pycocotools.** The bootstrap engine is compared with pycocotools on every evaluation; the largest difference is stored in `metrics.json` and a warning is logged if it exceeds 1e-6.
- **Intervals include training-seed variation.** The bootstrap resamples seeds as well as test images.
- **Every model is scored in the same class space.** 80-class models get a second NMS after truck→car and train→bus are merged, so they are not charged for duplicate boxes that 5-class students cannot produce.
- **Every run writes `config.json`** with its arguments, package versions and git commit.
