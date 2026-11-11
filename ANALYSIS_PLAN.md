# Analysis plan (pre-registration)

> **Status: DRAFT.** Finish it at the end of week 8 and **commit it before any
> held-out evaluation**. The commit hash goes in the paper. After that commit,
> nothing below may change.

## Data
- **Development:** ACDC fog val (100 images). Used freely during development.
- **Held-out 1:** ACDC fog train (~400 images; confirm the count: ____). Never evaluated before this commit.
- **Held-out 3:** Foggy Cityscapes validation cities (Frankfurt, Lindau, Münster), all three fog levels (____ images), built by `prep` and excluded from training. P1–P3 are tested on held-out 1 and 2; held-out 3 is reported alongside (its labels include fog-hidden objects, so it is not a pure test of real-fog transfer).
- **Held-out 2:** Foggy Driving (____ images, five classes confirmed: yes/no).
- **Class mapping:** person (+rider), bicycle, car (+truck), motorcycle, bus (+train).

## Primary metric
- COCO mAP@0.5, averaged over classes, computed by pycocotools (`wase.evaluate`, which reports the pycocotools numbers), imgsz 640.
- Secondary metrics: instance-weighted AP@0.5, mAP@0.75, mAP@[.5:.95], AP_S/M/L.

## Primary comparisons
| ID | Comparison | Seeds | Claim if confirmed |
|---|---|:-:|---|
| P1 | `naive` − `supervised` | 3 vs 3 | "Self-training on unlabelled foggy images transfers better (or worse) to real fog than fine-tuning on the human labels" |
| P2 | best refinement − `naive` | 3 vs 3 | "Confidence-gated DCP refinement is equivalent to (or worse than) plain pseudo-labelling" |
| P3 | best verification − `naive` (only if one is locked below) | 3 vs 3 | "Selecting pseudo-labels by agreement between views improves on (or does not improve on) plain pseudo-labelling" |

## Decision rules
- **Bootstrap:** 2,000 image resamples (seed 0). Inside each resample the training seeds of each
  condition are also drawn with replacement and then averaged, so intervals include seed-to-seed
  variation (`evaluate` default; `--fixed-seeds` is not used for any reported number).
- **P1 is confirmed** if the 95% CI of the difference excludes 0 on *both* held-out sets.
- **P2 "equivalent"** if the 90% CI lies within ±____ AP (TOST; planned ±0.01, to be fixed here from
  the interval width seen on *development* data, before this file is committed). "Worse" if the 95% CI is entirely below 0.
- **P3 is confirmed** under the same rule as P1.
- Oracle runs (`oracle-*`) use human labels of the training images. They are upper bounds, never
  compared as methods, and never used to choose the P3 variant on held-out data.
- **All other comparisons** are exploratory and labelled as such in the paper.
- Observed differences are reported alongside the bootstrap ones; every caption says which.

## Locked conditions
List the exact run names and the git commit used for each:

- naive: ____
- supervised: ____
- refinement variant for P2 (chosen on development data only): ____
- verification variant for P3 (chosen on development data only; "none" if nothing beat naive there): ____
