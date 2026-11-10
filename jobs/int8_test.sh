#!/bin/bash
#SBATCH --job-name=int8_test
#SBATCH --partition=C
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/kolanchi/thesis/runs/logs/%x_%j.out

source /home/kolanchi/thesis-env/bin/activate
cd /home/kolanchi/thesis/configs
mkdir -p /home/kolanchi/thesis/runs/int8

for v in 1269 1286 1287 1302 1303; do
  CK=$(ls /home/kolanchi/thesis/runs/burnscars_600m_baseline/lightning_logs/version_$v/checkpoints/best-epoch=*.ckpt)
  Q=/tmp/int8w_$v.ckpt
  echo "########## version $v : $CK"
  python /home/kolanchi/thesis/scripts/quantize_int8.py "$CK" "$Q"
  mv /tmp/int8w_${v}_int8_only.pt /home/kolanchi/thesis/runs/int8/
  echo "===== RESULT FP32 version $v"
  terratorch test --config burnscars_600m_baseline.yaml --ckpt_path "$CK" --trainer.accelerator=gpu --trainer.devices=1 --trainer.precision=32 --trainer.default_root_dir=/home/kolanchi/thesis/runs/int8/testlogs
  echo "===== RESULT INT8W version $v"
  terratorch test --config burnscars_600m_baseline.yaml --ckpt_path "$Q" --trainer.accelerator=gpu --trainer.devices=1 --trainer.precision=32 --trainer.default_root_dir=/home/kolanchi/thesis/runs/int8/testlogs
  rm -f "$Q"
done
echo "Finished $(date)"
