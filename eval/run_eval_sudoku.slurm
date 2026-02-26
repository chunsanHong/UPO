#!/bin/bash
#SBATCH --job-name=dpo
#SBATCH --nodes=1
#SBATCH --nodelist=hpc-pr-a-pod03
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=ultra.out
#SBATCH --error=ultra.err
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=hoarer@kaist.ac.kr

source /scratch/slurm-user3/anaconda3/etc/profile.d/conda.sh
conda activate d1_env

GPU_IDS=(0 1 2 3)

if [ $# -gt 0 ]; then
  GPU_IDS=("$@")
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}

echo "Using GPUs: $GPU_LIST    (num_processes=$NUM_GPUS)"

MASTER_PORT=29411                          
TASKS=("sudoku")

MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
EXTRA_PATH="/scratch/slurm-user3/chunsan/reproducible_upo/slurm_scripts/checkpoints/sudoku_pretrain+CE_1e-4_Temp0.8_1/checkpoint-3500/extra_head.pt"
out_dir="eval/sudoku_test"

CUDA_VISIBLE_DEVICES=$GPU_LIST \
srun accelerate launch \
  --main_process_port "$MASTER_PORT" \
  --num_processes "$NUM_GPUS" \
  eval.py \
    --dataset "sudoku" \
    --batch_size 1 \
    --output_dir "$out_dir" \
    --model_path "$MODEL_PATH" \
    --remasking "extra_head" \
    --extra_path "$EXTRA_PATH"