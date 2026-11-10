# WASE resubmission: the plan

*Written 22 Oct 2026. Target: IEEE RA-L (IROS 2027 option), deadline 1 March 2027.*

---

## 1. The original idea, in one paragraph

Annotating foggy data is slow and expensive. Rushed annotation is sloppy; careful annotation takes forever. WASE was meant to label foggy images automatically:

1. A **teacher** trained on clear images (COCO YOLOv8n) labels each foggy image.
2. Boxes it is unsure about (confidence roughly 0.1–0.5; the code used 0.10–0.65) are **de-fogged with DCP** and re-labelled by the same teacher.
3. The raw and refined label sets are **scored (DQS)**, and the winner becomes the label of the **raw** image.
4. A **student** trains on those labels.
5. The next teacher is **0.9 × old teacher + 0.1 × student**. Repeat for three generations; the last student is the final model.

**Result:** plain pseudo-labelling with no de-fogging ("naive") beat every WASE variant.

## 2. Why did naive win? Was it the fog, or the plan?

**It was the plan, not the fog.** Fog is actually the *best* case for DCP, because DCP is literally a model of fog. With rain, snow or night, DCP's physics doesn't apply at all, so switching weather would have made things worse.

Five design flaws, each enough on its own to stop WASE from beating naive:

| # | Flaw | Plain words | Status |
|---|---|---|---|
| 1 | **The scoring duel was rigged** | A refined box only replaced the raw one if it was *more confident*, and DQS = average confidence × √count. So the refined set almost always "won", whether its boxes were better or not | **Proven**: `python -m wase.selftest` (2,000 random trials) |
| 2 | **Refinement can't fix fog's main error** | In fog, the detector mostly *misses* objects. Refinement only re-scores boxes the teacher already found; it never adds a missed object or removes a false one. Labels are saved without confidence, so the only thing it could change was box geometry | By construction (`labeling.py`) |
| 3 | **Cropping + zooming breaks the detector** | A small crop blown up to 640 px looks like nothing the detector saw in training. It locks onto *parts* of the object, so boxes shrink (to ~0.5× area last round) while confidence rises | Measured last round. The new smoke run shows shrinking **even without DCP**, so it is a crop problem, not a de-fog problem (to confirm at full scale: study A2) |
| 4 | **DCP was used outside its assumptions** | DCP assumes big patches (15×15) and *one* fog colour for the whole picture. Crops used 3×3 patches and took the "fog colour" from the crop's brightest pixels, often a white car or headlight. And de-fogging can't add back detail fog destroyed (the data-processing inequality) | Theory + He et al. 2011 |
| 5 | **The 0.9/0.1 teacher mixing broke the loop** | The teacher has an 80-class answer layer, the student a 5-class one. Mixing left the old answer layer on a changed body, and that teacher found **zero** boxes. Each new student also had its class head rebuilt from scratch, and ~90% of its body reset to COCO | **Measured**: 36 head tensors (`model.22.cv3.*`) cannot transfer between 80 and 5 classes |

**Why naive works at all:**
- The student trains on foggy images, so its internal normalisation (BatchNorm) adapts to fog.
- Heavy augmentation lets the student generalise beyond its teacher (the "Noisy Student" effect).
- The teacher's labels only contain objects that are actually visible, so the student isn't asked to find the impossible.

That last point is the most interesting one scientifically (section 4).

## 3. Does last round's review still hold?

I re-checked everything I claimed, by measurement where possible, rather than asking you to trust me:

| Claim | How it was checked | Verdict |
|---|---|---|
| 1,918 of the "1,948 unlabelled" images have labels in the other folder | Read-only scan of the zip (`assessment_2026-10-21/checks/`); `prep.py` reproduces it (counter `city_sibling_other_folder`) | ✅ holds |
| The scene split leaks (1,432 / 2,975 scenes in both folders) | Same scan; `audit.json` reports `scenes_in_both_original_folders` | ✅ holds |
| Per-box visibility can be recovered from the three fog levels | Fit error 0.3% on real scenes; exact on synthetic fog (`selftest`) | ✅ holds |
| DQS can't judge label quality | Proven algebraically and tested (`selftest`) | ✅ holds |
| The old scorer differs from COCO (pooled classes, small-object penalty) | Code reading. The new engine matches pycocotools to 1e-16 (`selftest`) | ✅ holds |
| Teacher mixing re-initialises the head (36 tensors) | Measured with Ultralytics' own weight-transfer function | ✅ holds |
| The RTTS `.txt` files drop 28% of boxes but have the right class order | Scan of `RTTS/` | ✅ holds |
| ACDC fog *train* has detection labels (~400 images) | Not checkable until downloaded | ⚠️ verify after download |
| Which run produced the old 0.592 supervised score | Not checkable from the files here | ⚠️ unknown; the new baseline replaces it anyway |

One addition from your description: **the uncertain band.** You designed 0.1–0.5, but the code used 0.10–0.65. The new code makes it a setting (`--band 0.10,0.50`), so you can test your original design if you want.

## 4. The new core claim (working version)

> **Human labels in fog include objects the camera can no longer see; a detector's own labels don't.
> We measure that with fog physics and test whether it explains why self-made labels transfer to real fog better than human labels.**

This claim is safe to bet on because **every outcome is publishable**:

| If… | …then the paper says |
|---|---|
| Removing invisible human boxes closes the gap | "Invisible labels hurt; here's a physics-based filter that fixes it" (a practical recipe) |
| A lower learning rate closes the gap instead | "It's forgetting, not label quality" (a clean negative for the popular explanation) |
| The fixed baseline catches up with naive | "Self-training ≈ human labels; here's how many human labels it is worth" (the label-efficiency curve) |

The refinement work (Q2) becomes **one compact section**, "confidence is not a quality signal under fog": the DQS proof, the zoom study, and a few training runs.

## 5. What the new code does differently

| Problem last round | Fix in `wase/` |
|---|---|
| Label lookup lost 1,918 images | Labels found **by scene** across both folders (`prep.py`) |
| Naive and supervised trained on different images | One image list (`data/lists/train.txt`) for **every** condition |
| Same street in train and val | **Scene-level** split; `dev` = 10% of scenes |
| `best.pt` chosen by different rules | **No early stopping; `last.pt` always.** Validation uses human labels on dev images, only to monitor |
| Teacher/student weight mixing | Student promoted directly; class map taken from the running model |
| Shared label folders between runs | Each run has its own `data/` folder of image hardlinks and labels |
| Non-standard scorer | pycocotools for headline numbers; our engine (verified identical) for the bootstrap |
| Single seed, no equivalence tests | `--seed`; bootstrap averages seeds inside each resample; TOST equivalence with `--margin` |
| Hand-typed table numbers | `evaluate --report` builds every table from JSON |
| BBOX_ overlays in training | Excluded at prep |
| Unverified code | `python -m wase.selftest` must pass before any run |

## 6. Experiments

### 6.1 Data you still need

- **ACDC:** download `gt_detection_trainval.zip` and `rgb_anon_trainvaltest.zip` from <https://acdc.vision.ee.ethz.ch>. That gives fog val (100 images, development) and fog train (~400 images, held-out test, untouched).
- **Foggy Driving:** a second held-out set. Verify box annotations exist for our five classes; the fallback is DAWN fog.

### 6.2 Block A: mechanism, no training (≈ a few hours)

Run each on ACDC fog val with two teachers: the COCO model, and the `naive` seed-0 final model.

| Study | Command | Answers |
|---|---|---|
| A1–A3 | `diagnose scale` | Does box shrinkage grow with zoom? Does it happen without DCP? Is the confidence gain real, or only the keep-if-higher selection? |
| A4 | `diagnose selection` | When DQS picks the refined layout, is that layout actually better? |
| A5 | `diagnose calibration` | Are confidences calibrated (ECE)? |
| A6 | `diagnose agreement --split train` | Teacher labels vs human labels, by size, **visibility** and source |

**Gate A:** read these results with your supervisor before any refinement training.

### 6.3 Block Q1: human vs automatic labels (main question)

| Run | Seeds | What it tests |
|---|:-:|---|
| `naive` | 3 | Main comparison, automatic side |
| `supervised` | 3 | Main comparison, human side (fixed baseline) |
| `supervised --label-filter visibility --tau 0.1` | 1→3 | Do invisible labels cause the harm? |
| `supervised --label-filter size --min-px 12` | 1 | Is it size, rather than visibility? |
| `supervised --label-filter teacher` | 1 | Do automatic labels win by *filtering*, or by drawing better boxes? |
| `supervised --source rtts` / `--source city` | 1 + 1 | Is the harm concentrated in the synthetic half? |
| `supervised --lr-scale 0.1` | 1 | Forgetting explanation |
| `supervised --imgsz 1024` | 1 | Is resolution the cause? (Add `naive --imgsz 1024` if it helps) |
| `supervised --fraction 0.1 / 0.25 / 0.5` | 1 each | Label-efficiency curve |
| `naive --generations 1 --epochs 120` | 1 | Do three generations matter? |

**Gate 1 (after seed 0 of `naive` and `supervised`):** if the fixed gap is under ~0.02 AP, switch the framing to "how many human labels is self-training worth" (section 4, third row).

### 6.4 Block Q2: refinement (trimmed)

| Run | Seeds |
|---|:-:|
| `crop-dqs` | 3 |
| `inplace-dqs` | 3 |
| `rawflip-wbf` (two views, no enhancement) | 1 |
| `global-dcp` | 1 |
| `crop-dqs-nogate`, `crop-wbf`, `inplace-wbf` | only if time remains |

The primary Q2 comparison is **best refinement − naive, tested for equivalence (±0.01 AP)**. Comparing two variants that both lost to naive would prompt a "so what?" from reviewers.

### 6.5 Block C: held-out confirmation

1. Fill in `ANALYSIS_PLAN.md` and **commit it**.
2. Lock all conditions.
3. Evaluate everything **once** on ACDC fog train and Foggy Driving.
4. Report development and held-out results side by side.

### 6.6 Compute

About 25 runs. Each three-generation run is ~120 epochs on ~11.9k images: roughly 4–6 h on the RTX 4080 SUPER, and several times that on the RTX 2060 here. Total **≈ 125 GPU-hours** on the 4080.

## 7. Human checkpoints (nothing continues until a person signs off)

The code has self-tests, but a person must check that it does what we think on the real data.

| # | When | What a person checks | Stop if |
|---|---|---|---|
| 1 | After `prep` | `data/audit.json`: 13,247 real images, 30 excluded for having no label, 1,918 labels found via `city_sibling_other_folder`, 41,203 RTTS boxes. Then open `results/inspect/.../index.html` (`python -m wase.inspect_labels`, 50 random images) | any count differs, or any box is visibly misplaced or misclassified |
| 2 | After downloading ACDC | `evaluate` the COCO model on ACDC fog val; the GT loader reports images/objects (expect 100 / ~637) | not near the old 0.618, or the object count is off |
| 3 | After `naive` + `supervised` seed 0 | naive near the old 0.716; open `gen0_labels.json`; draw the teacher's labels with `inspect_labels --teacher` | naive far from 0.716 |
| 4 | After Block A | read the diagnose JSONs with your supervisor | — (this is the plan's decision point) |

## 8. Order of work

| Week | Work |
|---|---|
| 1 | `selftest` → `prep` (full) → check `audit.json` → download ACDC → `evaluate` the COCO model on ACDC fog val (it may move a little from 0.618 under the standard scorer) |
| 2 | `naive` + `supervised`, seed 0 → **Gate 1** |
| 3 | Block A → **Gate A**, review with supervisor |
| 4–8 | Q1 mechanism runs, seeds 1–2 for the main pairs, label-efficiency curve, trimmed Q2 |
| 9 | Write `ANALYSIS_PLAN.md`, commit, run held-out **once** |
| 10–18 | Writing, `evaluate --report`, internal review |

## 9. Rules

- `python -m wase.selftest` must pass before any run that will be reported.
- Never type a number into a table; use `evaluate --report`.
- Commit after every completed run. `runs/**/config.json`, `gen*_labels.json` and `results/**/metrics.json` are tracked by git; weights are not, so back them up off the machine.
- If a result contradicts the story, report it plainly. Don't tune toward a preferred outcome.
- Don't touch the held-out sets until `ANALYSIS_PLAN.md` is committed.

---

## 10. Version 2 additions (branch `wasev2`, 10 Nov 2026)

Added after reading the reviews of the previous submission. Sections 1-9 are unchanged; where
this section disagrees with them, this section wins.

### 10.1 What the reviewers objected to, and what answers it

| Reviewer point | Answer in this branch |
|---|---|
| Ablations changed two things at once (gate + placement; fusion + oversampling) | `crop-dqs-nogate`, `crop-wbf` and `inplace-wbf` move from "if time remains" into block Q2, so every step changes one factor. Oversampling stays off |
| Padding changes context and scale together | `diagnose scale` at s=1 removes context at constant scale. `selftest` now shows a zero-padding crop can only return a box inside the raw box, so shrinkage is partly geometric, not only a detector failure. Say so in the paper |
| Confidence gain was conditional on acceptance | Unconditional change is recorded (`mean_dconf_unconditional`, study A3) |
| DQS judged on total counts, not per image | Study A4 scores each chosen layout against ground truth |
| Supervised comparison not matched; resolution cause untested | One image list and schedule for all; `--imgsz 1024`, size and teacher filters (block Q1) |
| Causal explanation asserted, not tested | **Oracle decomposition** (10.2) replaces explanation by measurement |
| Single seed; image bootstrap ignores training variability | 3 seeds for primary pairs, and the bootstrap now **resamples training seeds as well as images** (`evaluate`, default) |
| Result tuned and tested on the same 100 images | Held-out sets (block C), touched once |
| "Better calibration" claimed from thresholds | ECE is measured (study A5) |
| Diagnosis only, no method; refinement still loses to naive | **Verification conditions** (10.3) and a **label-time teacher** (10.4): two ways to change which objects are labelled |
| One detector, one enhancement, one weather | See 10.6. Not solved by code alone |
| Related work ends in 2023 | See 10.7 |

### 10.2 Oracle decomposition: which label error actually costs accuracy?

Refinement only ever moved boxes. Before building anything else, measure which kind of pseudo-label
error is worth fixing. Each run is naive self-training with **one** error type repaired from the
human labels (matching: same class, IoU >= 0.5, one-to-one):

| Run | Repair | If it closes the gap to the best condition |
|---|---|---|
| `oracle-nofp` | false boxes removed | precision is the bottleneck -> rejection (verify, higher thresholds) can pay |
| `oracle-addfn` | missed objects added | recall is the bottleneck -> admission (verify, stronger teacher) can pay |
| `oracle-geom` | matched boxes get human geometry | localisation is the bottleneck -> only then is box refinement worth another look |

These are upper bounds, not methods, and are labelled so in every table. Caveats to state: the human
labels are themselves imperfect (the Cityscapes label files have no `train` class, so a correctly
detected tram counts as "false"; tiny and fog-hidden human boxes count as "missed"). Read
`gen0_labels.json` (`oracle_matched / oracle_false / oracle_missed`) before training to see how
large each error pool is.

### 10.3 Verification: agreement between views, not confidence

The original uncertain band never rejected anything: a box that failed re-detection was kept. The
`verify-*` conditions use the second view to decide and never touch geometry:

- candidates are taken down to the lower band edge (0.10, below the teacher threshold);
- above the upper edge (0.65): kept;
- inside the band: kept only if the second view has a same-class box with IoU >= 0.5.

Second views: `verify-flip` (mirror), `verify-dcp` (full-frame DCP), `verify-scale` (1.5x inference
size). `--verify-mode admit|reject` switches off one half to show which half matters. The control
for "it is only the lower threshold" is `naive --teacher-conf 0.10`.

`verify-dcp` is the honest descendant of WASE: de-fogging used as a second opinion rather than as a
box editor. If `verify-flip` does as well, the paper says de-fogging is not what helps, and that is
still a result.

### 10.4 Label-time teacher

`--teacher yolov8m.pt --label-imgsz 1280 --student yolov8n.pt --generations 1 --epochs 120`: a larger
model at higher resolution writes the labels once; the deployed model is the same YOLOv8n at 640.
This is the most likely way to raise the absolute numbers and the least novel, so it is reported as
a strong baseline that any proposed method must be compared with, not as the contribution.
(`--label-imgsz` alone, with the default teacher, separates the effect of resolution from model size.)

### 10.5 BatchNorm-only baseline

Section 2 credits part of naive's gain to normalisation layers adapting to fog. `python -m wase.adabn`
replaces the COCO model's BatchNorm statistics with those of the foggy training images and changes
nothing else. It costs minutes and turns that sentence into a number.

### 10.6 Scope (not solved by this branch)

The strongest reviewer said: one detector, one enhancement, one weather, 100 images. In order of
value per GPU-hour:

1. **More evaluation images at no training cost:** ACDC rain, snow and night use the same label
   format; `evaluate` runs on them unchanged. Report them as "does the fog-trained model transfer",
   not as a claim about those conditions.
2. **A second detector size** (`--teacher yolov8s.pt --student yolov8s.pt`) for the two primary pairs.
3. **A standard benchmark:** Cityscapes -> Foggy Cityscapes with the usual 8 classes, so numbers sit
   beside published ones. Needs a second class table in `common.py` (not written yet).
4. **A learned de-hazer** next to DCP for `global-dcp` and `verify-dcp` (not written yet).

### 10.7 Reading to do before any novelty claim

- ADT (Wang et al., 2024, arXiv:2403.15786): scale changes during pseudo-labelling in poor visibility.
- Foggy-Aware Teacher (Qin et al., RA-L 2025): de-hazing before the teacher. Same target venue.
- Soft Teacher (box jitter to select regression targets) and Probabilistic Teacher (explicit
  localisation uncertainty): both are quality signals other than classification confidence, so the
  sentence "the quality signal is still confidence" must go.
- Consistency / agreement-based pseudo-label selection in semi-supervised detection: `verify-*` is
  close to this family. **Check how close before calling it new.** The defensible claim may be the
  measurement (which error matters under fog, 10.2) with verification as its consequence.

### 10.8 Corrections to sections 2 and 4

- **Flaw 3** ("crop + zoom breaks the detector"): with zero padding the re-detected box must lie
  inside the raw box, so an area ratio below 1 is guaranteed. The detector failure is real only to
  the extent that shrinkage exceeds what s=1 (no zoom) already shows.
- **Flaw 4** (data-processing inequality): drop this argument. Pre-processing cannot add information,
  but it can help a fixed, mismatched detector; whether it does is empirical.
- **Core claim** (section 4): `data/audit.json` shows boxes with transmittance below 0.1 are 0%,
  0.38% and 7.93% of Foggy Cityscapes boxes at the three fog levels, about 2% of all human boxes;
  RTTS has no transmittance. A 2% label change is unlikely to carry the gap alone. Keep visibility as
  one hypothesis beside size (30.6% of boxes are under 12 px at 640) and the missing `train` class,
  and let 10.2 say which pool of errors matters.
- **Checkpoints 2 and 3** ("near the old 0.618 / 0.716"): the old scorer pooled classes; the headline
  is now the class mean. Compare the old numbers with `instance_weighted_AP50`, not `mAP50`.
- **Equivalence margin** (+-0.01 AP): check the width of the bootstrap interval after the first two
  runs. If it is wider than the margin on the held-out sets, widen the margin in `ANALYSIS_PLAN.md`
  *before* committing it, or the test can only ever say "inconclusive".
- **`--lr-scale 0.1`** is run for `supervised` and for `naive`, otherwise "forgetting" cannot be
  told apart from "a lower learning rate helps everything".

### 10.9 Revised order

| Step | Work | Decision |
|---|---|---|
| 1 | `selftest`, `prep`, `naive` + `supervised` seed 0, `adabn` (block `week2`) | Gate 1 as before |
| 2 | Block A diagnostics | Gate A as before |
| 3 | Block `oracle` (3 runs) | Which of precision / recall / localisation has headroom |
| 4 | Block `verify` and block `teacher`, seed 0 | Does anything beat naive on development data? |
| 5 | Q1 mechanism runs, full Q2 ablation, seeds 1-2 for every primary pair | |
| 6 | Fill and commit `ANALYSIS_PLAN.md`; held-out once | |

