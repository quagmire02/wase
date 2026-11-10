# End-to-end smoke test on a tiny subset (~10 min on an RTX 2060).
# Numbers it produces are meaningless; it only proves every stage runs.
# Run from the project root:  powershell -ExecutionPolicy Bypass -File scripts\smoke.ps1
$ErrorActionPreference = "Stop"
$py = if ($env:WASE_PYTHON) { $env:WASE_PYTHON } else { "python" }
$D = "smoke\data"; $R = "smoke\runs"; $O = "smoke\results"

& $py -m wase.selftest;                                                        if ($LASTEXITCODE) { exit 1 }
& $py -m wase.prep --data $D --limit-scenes 15;                                if ($LASTEXITCODE) { exit 1 }
foreach ($c in "naive", "crop-dqs", "inplace-wbf", "verify-flip", "oracle-addfn") {
    & $py -m wase.train --condition $c --data $D --runs $R --generations 2 --epochs 2 `
        --teacher-conf 0.15,0.005 --batch 8 --workers 0;                       if ($LASTEXITCODE) { exit 1 }
}
& $py -m wase.train --condition supervised --label-filter visibility --tau 0.2 `
    --data $D --runs $R --generations 1 --epochs 1 --batch 8 --workers 0;      if ($LASTEXITCODE) { exit 1 }

& $py -m wase.adabn --data $D --runs $R --max-images 32;                        if ($LASTEXITCODE) { exit 1 }

$G = @("--gt", "$D\dev_gt.json", "--img-root", "$D\store\images")
& $py -m wase.evaluate --set smoke_dev --results $O @G `
    --models "$R\naive*\seed*\weights\final.pt" "$R\supervised*\seed*\weights\final.pt" `
    --compare "supervised_vis0.2_1gen_ep1:naive_2gen_ep2_conf0.15-0.005" --margin 0.01 --bootstrap 200
                                                                               if ($LASTEXITCODE) { exit 1 }
& $py -m wase.evaluate --report --results $O;                                  if ($LASTEXITCODE) { exit 1 }
foreach ($s in "scale", "selection") {
    & $py -m wase.diagnose $s @G --out "$O\diagnose" --tag smoke;              if ($LASTEXITCODE) { exit 1 }
}
& $py -m wase.diagnose agreement --data $D --split train --out "$O\diagnose" --tag smoke
                                                                               if ($LASTEXITCODE) { exit 1 }
Write-Host "`nSMOKE TEST PASSED -- see $O\tables.md"
