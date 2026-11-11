# Every run in PLAN.md section 6, in the order of section 7.
# Each block is safe to re-run: finished generations are skipped.
# Stop at the gates and read the results before continuing.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 -Block week2
param([Parameter(Mandatory = $true)]
      [ValidateSet("prep", "week2", "blockA", "oracle", "verify", "teacher", "q1", "seeds", "q2", "optional")] [string]$Block,
      [string]$AcdcGt = $env:ACDC_FOG_VAL_GT, [string]$AcdcImg = $env:ACDC_FOG_VAL_IMG,
      # -Pilot: one generation of 40 epochs, to RANK conditions on development data at about a third
      # of the cost. Pilot runs get their own names (..._1gen_ep40) and are never reported as results.
      [switch]$Pilot)
$ErrorActionPreference = "Stop"
$py = if ($env:WASE_PYTHON) { $env:WASE_PYTHON } else { "python" }
function T([string[]]$a) { if ($Pilot) { $a += @("--generations", "1", "--epochs", "40") }
                           & $py -m wase.train @a; if ($LASTEXITCODE) { throw "train failed: $a" } }

switch ($Block) {
  "prep"   { & $py -m wase.selftest; & $py -m wase.prep }

  # Gate 1: compare these two on ACDC fog val before going further
  "week2"  { T "--condition", "naive", "--seed", "0"
             T "--condition", "supervised", "--seed", "0"
             & $py -m wase.adabn; if ($LASTEXITCODE) { throw "adabn failed" } }

  # Gate A: mechanism studies, no training
  "blockA" { if (-not $AcdcGt) { throw "set -AcdcGt / -AcdcImg (ACDC fog val)" }
             $G = @("--gt", $AcdcGt, "--img-root", $AcdcImg)
             foreach ($t in @(@("coco", "..\yolov8n.pt"), @("naive", "runs\naive\seed0\weights\final.pt"))) {
               & $py -m wase.diagnose scale     @G --weights $t[1] --tag $t[0]
               & $py -m wase.diagnose selection @G --weights $t[1] --tag $t[0] }
             & $py -m wase.diagnose calibration @G --tag acdc --models coco=..\yolov8n.pt `
                 naive=runs\naive\seed0\weights\final.pt supervised=runs\supervised\seed0\weights\final.pt
             & $py -m wase.diagnose agreement --split train --tag coco_gen0
             & $py -m wase.diagnose errors @G --tag acdc --models coco=..\yolov8n.pt `
                 naive=runs\naive\seed0\weights\final.pt supervised=runs\supervised\seed0\weights\final.pt }

  # PLAN 10.2: which pseudo-label error costs accuracy? (upper bounds, 1 seed each)
  "oracle" { foreach ($c in "oracle-nofp", "oracle-addfn", "oracle-geom") { T "--condition", $c, "--seed", "0" } }

  # PLAN 10.3: agreement between views instead of confidence (seed 0; more seeds for the best one)
  "verify" { foreach ($c in "verify-flip", "verify-dcp", "verify-scale") { T "--condition", $c, "--seed", "0" }
             T "--condition", "naive", "--teacher-conf", "0.10"            # control: lower threshold only
             foreach ($m in "admit", "reject") { T "--condition", "verify-flip", "--verify-mode", $m } }

  # PLAN 10.4: strong teacher at label time, same small deployed model
  "teacher" { $one = @("--generations", "1", "--epochs", "120")
              T (@("--condition", "naive", "--label-imgsz", "1280") + $one)
              T (@("--condition", "naive", "--teacher", "yolov8m.pt", "--student", "yolov8n.pt") + $one)
              T (@("--condition", "naive", "--teacher", "yolov8m.pt", "--student", "yolov8n.pt",
                   "--label-imgsz", "1280") + $one) }

  # Q1 mechanism runs (1 seed each)
  "q1"     { T "--condition", "supervised", "--label-filter", "visibility", "--tau", "0.1"
             T "--condition", "supervised", "--label-filter", "size", "--min-px", "12"
             T "--condition", "supervised", "--label-filter", "teacher"
             T "--condition", "supervised", "--source", "rtts"
             T "--condition", "supervised", "--source", "city"
             T "--condition", "supervised", "--lr-scale", "0.1"
             T "--condition", "naive", "--lr-scale", "0.1"
             T "--condition", "supervised", "--imgsz", "1024"
             foreach ($f in "0.1", "0.25", "0.5") { T "--condition", "supervised", "--fraction", $f }
             T "--condition", "naive", "--generations", "1", "--epochs", "120" }

  # Extra seeds for the primary comparisons
  "seeds"  { foreach ($s in 1, 2) { foreach ($c in "naive", "supervised", "crop-dqs", "inplace-dqs") {
               T "--condition", $c, "--seed", "$s" } } }

  # Q2 refinement (seed 0; seeds 1-2 come from the "seeds" block). The one-factor chain:
  #   crop-dqs-nogate -> crop-dqs (adds the gate) -> inplace-dqs (changes placement) -> inplace-wbf (changes fusion)
  "q2"     { foreach ($c in "crop-dqs-nogate", "crop-dqs", "inplace-dqs", "inplace-wbf", "crop-wbf",
                             "rawflip-wbf", "global-dcp") { T "--condition", $c, "--seed", "0" } }

  # Only if time remains
  "optional" { T "--condition", "naive", "--imgsz", "1024"
               T "--condition", "crop-dqs", "--band", "0.10,0.50" }
}
