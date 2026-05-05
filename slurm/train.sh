#!/bin/bash
#SBATCH --job-name=param_golf
#SBATCH --output=/home/spatadia/parameter-golf/logs/%x_%j.out
#SBATCH --error=/home/spatadia/parameter-golf/logs/%x_%j.err
#SBATCH --partition=academic
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --time=08:00:00

set -e

# Capture SCRIPT before .env can override it
_SCRIPT="${SCRIPT:-train_gpt.py}"

module load cuda12.6/toolkit/12.6.2
module load python/3.11.7/lty5ef6
source ~/parameter-golf/parameter-golf/.venv/bin/activate
cd ~/parameter-golf/parameter-golf

# Load secrets (.env may contain a SCRIPT variable — use _SCRIPT instead)
set -a; source ~/parameter-golf/parameter-golf/.env; set +a

export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=500
export WANDB_ENABLED=1

python -m torch.distributed.run --standalone --nproc_per_node=2 "$(pwd)/$_SCRIPT"
