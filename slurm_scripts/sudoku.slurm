#!/bin/bash
#SBATCH --job-name=dpo
#SBATCH --nodes=1
#SBATCH --nodelist=hpc-pr-a-pod03

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --output=out/ultra.out
#SBATCH --error=out/ultra.err
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=hoarer@kaist.ac.kr

source /scratch/slurm-user3/anaconda3/etc/profile.d/conda.sh
conda activate d1_env

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WANDB_PROJECT=...
export WANDB_ENTITY=...
export WANDB_API_KEY=...

DATASET="sudoku"
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
PRETRAIN_MODEL_PATH=...
RANDOM_MODEL_PATH=...

for i in {1..5}
do
    RUN_NAME=${DATASET}_pretrain+CE_1e-4_Temp0.8_${i}

    srun accelerate launch \
        --config_file ./accelerate_a100.yaml \
        --main_process_port 24445 ../upo_train.py \
        --config ./train_sudoku.yaml \
        --model_path $MODEL_PATH \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --output_dir checkpoints/$RUN_NAME \
        --sync_ref_model false \
        --beta 0.0001 \
        --remasking "extra_head" \
        --extra_model_path $PRETRAIN_MODEL_PATH \
        --divergence "CE" \
        --extra_bim_size 5 \
        --extra_temperature 0.05 \
        --temperature 0.8


    RUN_NAME=${DATASET}_pretrain_wo_reg_Temp0.8_${i}

    srun accelerate launch \
        --config_file ./accelerate_a100.yaml \
        --main_process_port 24445 ../upo_train.py \
        --config ./train_sudoku.yaml \
        --model_path $MODEL_PATH \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --output_dir checkpoints/$RUN_NAME \
        --sync_ref_model false \
        --beta 0.0000000000001 \
        --remasking "extra_head" \
        --extra_model_path $PRETRAIN_MODEL_PATH \
        --divergence "CE" \
        --extra_bim_size 5 \
        --extra_temperature 0.05 \
        --temperature 0.8

    RUN_NAME=${DATASET}_pretrain+KL_1e-4_Temp0.8_${i}

    srun accelerate launch \
        --config_file ./accelerate_a100.yaml \
        --main_process_port 24445 ../upo_train.py \
        --config ./train_sudoku.yaml \
        --model_path $MODEL_PATH \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --output_dir checkpoints/$RUN_NAME \
        --sync_ref_model false \
        --beta 0.0001 \
        --remasking "extra_head" \
        --extra_model_path $PRETRAIN_MODEL_PATH \
        --divergence "KL" \
        --extra_bim_size 5 \
        --extra_temperature 0.05 \
        --temperature 0.8

    RUN_NAME=${DATASET}_Random_Temp0.8_${i}

    srun accelerate launch \
        --config_file ./accelerate_a100.yaml \
        --main_process_port 24445 ../upo_train.py \
        --config ./train_sudoku.yaml \
        --model_path $MODEL_PATH \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --output_dir checkpoints/$RUN_NAME \
        --sync_ref_model false \
        --beta 0.000000000001 \
        --remasking "extra_head" \
        --extra_model_path $RANDOM_MODEL_PATH \
        --divergence "CE" \
        --extra_bim_size 5 \
        --extra_temperature 0.05 \
        --temperature 0.8

    RUN_NAME=${DATASET}_Random_TopK_KL_1e-4_Temp0.8_${i}

    srun accelerate launch \
        --config_file ./accelerate_a100.yaml \
        --main_process_port 24445 ../upo_train.py \
        --config ./train_sudoku.yaml \
        --model_path $MODEL_PATH \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --output_dir checkpoints/$RUN_NAME \
        --sync_ref_model false \
        --beta 0.0001 \
        --remasking "extra_bim" \
        --extra_model_path $RANDOM_MODEL_PATH \
        --divergence "KL" \
        --extra_bim_size 5 \
        --extra_temperature 0.05 \
        --temperature 0.8
done