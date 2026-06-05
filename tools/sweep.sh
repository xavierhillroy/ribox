#!/bin/bash
#SBATCH --job-name=lgp_sweep
#SBATCH --array=0-149              # 6 targets × 25 seeds = 150 cells (0..149)
#SBATCH --time=00:15:00            # generous; each cell is seconds-to-minutes
#SBATCH --mem=512M                 # runs are tiny
#SBATCH --cpus-per-task=1          # engine is single-threaded
#SBATCH --output=logs/%A_%a.out    # %A = array id, %a = task id
#SBATCH --account=def-YOURPI       # <-- set your DRAC allocation account

set -euo pipefail

targets=(nguyen1 nguyen2 nguyen3 nguyen4 nguyen5 nguyen6)
seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)

n_seeds=${#seeds[@]}
t_idx=$(( SLURM_ARRAY_TASK_ID / n_seeds ))   # -> target
s_idx=$(( SLURM_ARRAY_TASK_ID % n_seeds ))   # -> seed

target=${targets[$t_idx]}
seed=${seeds[$s_idx]}

mkdir -p results programs history logs
echo "cell ${SLURM_ARRAY_TASK_ID}: target=${target} seed=${seed} on $(hostname)"

./lgp_run \
    datasets/${target}_train.csv \
    datasets/${target}_test.csv \
    ${target} \
    ${seed} \
    results/${target}_seed${seed}.csv \
    programs/${target}_seed${seed}.txt \
    history/${target}_seed${seed}.csv