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
