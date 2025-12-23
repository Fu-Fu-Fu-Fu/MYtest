#!/bin/bash
#SBATCH -J fl_test         # 作业名
#SBATCH --partition=gpu       # 分区
#SBATCH -N 1                  # 节点数
#SBATCH --gres=gpu:a40:1     # GPU 数
#SBATCH --cpus-per-task=4    # CPU 核心数
#SBATCH --time=2-00:00:00     # 运行时长
#SBATCH --mem=64G            # 内存
#SBATCH -o train.log  # 合并输出文件
#SBATCH -e train.log   # 错误也写到同一个文件

# ===== 记录开始时间 =====
start_time=$(date +%s)

cleanup () {
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    echo "任务在 $(date) 结束或中断 (hostname=$(hostname))"
    echo "总耗时: $((runtime / 3600)) 小时 $(((runtime % 3600) / 60)) 分钟 $((runtime % 60)) 秒"
}
trap cleanup EXIT   # 捕捉退出（正常/异常都能输出）

echo "任务开始运行于 $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "正在使用的显卡信息："
nvidia-smi --query-gpu=name --format=csv,noheader

set -x
python -u src/train_rl.py \
    --total_episodes 5000 \
    --max_steps 20 \
    --n_candidates 128 \
    --hidden_dim 64 \
    --n_cross_attn_layers 2 \
    --update_every 20 \
    --save_every 1000 \
    --save_dir ./runs/ppo_bo_1221 \
    --seed 42