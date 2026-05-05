#!/bin/bash
#SBATCH --job-name=fineweb_download
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --partition=academic
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -e

module load cuda12.6/toolkit/12.6.2
module load python/3.11.7/lty5ef6
source ~/parameter-golf/.venv/bin/activate
cd ~/parameter-golf/parameter-golf

python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
echo "Done."
