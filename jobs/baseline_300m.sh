#!/bin/bash
#SBATCH --job-name=bs300_base
#SBATCH --partition=C
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/kolanchi/thesis/runs/logs/%x_%j.out

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "GPU given by Slurm: $CUDA_VISIBLE_DEVICES"
nvidia-smi

source /home/kolanchi/thesis-env/bin/activate
cd /home/kolanchi/thesis/configs

terratorch fit --config burnscars_300m_baseline.yaml --trainer.accelerator=gpu --trainer.devices=1 "$@"

echo "Finished $(date)"
