#!/bin/bash
#SBATCH -J HELLO
#SBATCH -p your partition
#SBATCH --quotatype=reserved
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1           
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=finetune_7b_multi_32.out



export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

NODE_RANK=$SLURM_NODEID
NUM_NODES=$SLURM_NNODES
WORLD_SIZE=$SLURM_NTASKS

MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT



# Training entry point
entry_file=qwenvl/train/train_qwen.py


# Launch training
srun --ntasks-per-node=1 --gres=gpu:8 --kill-on-bad-exit=1 bash -lc '
  conda activate qwen
  cd ./Qwen2.5-VL/qwen-vl-finetune
  echo "[`hostname`] SLURM_NODEID=$SLURM_NODEID  Starting torchrun ..."
  llm=Qwen/Qwen2.5-VL-7B-Instruct
  deepspeed=./scripts/zero3.json

  datasets=physxnet%25,physxmobility%25
  run_name="qwen2vl-baseline_7b_32"
  output_dir=./output_7b_32
  lr=2e-5
  batch_size=1
  grad_accum_steps=4

  args=(
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 262144 \
    --min_pixels 65536 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb
    )


  exec torchrun \
    --nproc_per_node=8 \
    --nnodes='"$SLURM_NNODES"' \
    --node_rank=$SLURM_NODEID \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    "'"$entry_file"'" "${args[@]}"
'

