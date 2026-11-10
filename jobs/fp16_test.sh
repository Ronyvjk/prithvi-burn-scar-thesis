#!/bin/bash
#SBATCH --job-name=fp16_test
#SBATCH --partition=C
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/kolanchi/thesis/runs/logs/%x_%j.out

source /home/kolanchi/thesis-env/bin/activate
cd /home/kolanchi/thesis/configs
OUT=/home/kolanchi/thesis/runs/int8/testlogs

for CK in /home/kolanchi/thesis/runs/burnscars_300m_baseline/lightning_logs/version_*/checkpoints/best-epoch=*.ckpt; do
  v=$(echo $CK | grep -o "version_[0-9]*" | cut -d_ -f2)
  echo "===== RESULT FP32_300M version $v"
  terratorch test --config burnscars_300m_baseline.yaml --ckpt_path "$CK" --trainer.accelerator=gpu --trainer.devices=1 --trainer.precision=32 --trainer.default_root_dir=$OUT
  echo "===== RESULT FP16_300M version $v"
  terratorch test --config burnscars_300m_baseline.yaml --ckpt_path "$CK" --trainer.accelerator=gpu --trainer.devices=1 --trainer.precision=16-true --trainer.default_root_dir=$OUT
done

for v in 1269 1286 1287 1302 1303; do
  CK=$(ls /home/kolanchi/thesis/runs/burnscars_600m_baseline/lightning_logs/version_$v/checkpoints/best-epoch=*.ckpt)
  echo "===== RESULT FP16_600M version $v"
  terratorch test --config burnscars_600m_baseline.yaml --ckpt_path "$CK" --trainer.accelerator=gpu --trainer.devices=1 --trainer.precision=16-true --trainer.default_root_dir=$OUT
done
echo "Finished $(date)"