#!/bin/bash
#SBATCH --job-name=lowbit300
#SBATCH --partition=C
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/kolanchi/thesis/runs/logs/%x_%j.out

source /home/kolanchi/thesis-env/bin/activate
cd /home/kolanchi/thesis/configs
OUT=/home/kolanchi/thesis/runs/int8/testlogs

for b in 8 4 3; do
  for CK in /home/kolanchi/thesis/runs/burnscars_300m_baseline/lightning_logs/version_*/checkpoints/best-epoch=*.ckpt; do
    v=$(echo $CK | grep -o "version_[0-9]*" | cut -d_ -f2)
    Q=/tmp/q300_${b}_$v.ckpt
    python /home/kolanchi/thesis/scripts/quantize_int8.py "$CK" "$Q" $b
    echo "===== RESULT INT${b}W_300M version $v"
    terratorch test --config burnscars_300m_baseline.yaml --ckpt_path "$Q" --trainer.accelerator=gpu --trainer.devices=1 --trainer.precision=32 --trainer.default_root_dir=$OUT
    rm -f "$Q" "${Q%.ckpt}_int_only.pt"
  done
done
echo "Finished $(date)"