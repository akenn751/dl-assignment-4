$base = "artifacts"
$seed = 123
$seed_text = "The "
$lr = 5e-4
$save_every = 1
$breakpoints = 1
$sample_length = 200

$hiddens = @(32,64,128)
$seqs = @(25,50,100)
$batches = @(32,64)

foreach ($h in $hiddens) {
  foreach ($s in $seqs) {
    foreach ($b in $batches) {
      $name = "quick_h${h}_s${s}_b${b}"
      Write-Host "Running $name"
      python -m src.train --epochs 1 --batch $b --seq $s --hidden $h --lr $lr --exp-name $name --save-dir $base --save-every $save_every --breakpoints $breakpoints --sample-length $sample_length --seed $seed --seed-text "$seed_text" --max-steps 500
      Write-Host "Finished $name. Artifacts: $base\logs\$name  $base\plots\$name  $base\checkpoints\$name"
    }
  }
}