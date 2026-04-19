# run_overnight_top3_simple.ps1
$base = "artifacts"
$seed = 123
$seed_text = "The "
$lr = 5e-4
$save_every = 1
$breakpoints = 1
$sample_length = 400
$epochs = 12

# final jobs (run sequentially)
$jobs = @(
  @{ name="final_h512_s150_b64"; hidden=512; seq=150; batch=64 },
  @{ name="final_h256_s200_b64"; hidden=256; seq=200; batch=64 },
  @{ name="final_h256_s150_b32"; hidden=256; seq=150; batch=32 }
)

foreach ($job in $jobs) {
  $name = $job.name
  $h = $job.hidden
  $s = $job.seq
  $b = $job.batch

  Write-Host "`n=== Running $name ==="
  python -m src.train --epochs $epochs --batch $b --seq $s --hidden $h --lr $lr --exp-name $name --save-dir $base --save-every $save_every --breakpoints $breakpoints --sample-length $sample_length --seed $seed --seed-text "$seed_text"
  Write-Host "Finished $name. Artifacts: $base\logs\$name  $base\plots\$name  $base\checkpoints\$name"
}
