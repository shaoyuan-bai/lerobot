#!/bin/bash
# RM65 数据录制脚本
# 使用方法: bash record_rm65.sh

# 设置你的 Hugging Face 用户名
# HF_USER=$(huggingface-cli whoami | head -n 1)
HF_USER="woosh"  # 修改为你的用户名

# 数据集名称
DATASET_NAME="rm65_demo_$(date +%Y%m%d_%H%M%S)"

# 任务描述
TASK="Pick and place task with RM65 dual arm robot"

# 录制参数
NUM_EPISODES=10        # 录制集数
EPISODE_TIME_S=30      # 每集录制时间(秒)
RESET_TIME_S=10        # 重置时间(秒)
FPS=30                 # 帧率

echo "============================================================"
echo "RM65 数据录制"
echo "============================================================"
echo "数据集: ${HF_USER}/${DATASET_NAME}"
echo "任务: ${TASK}"
echo "录制 ${NUM_EPISODES} 集, 每集 ${EPISODE_TIME_S} 秒"
echo "============================================================"

# 提示:RM65使用拖动示教,不需要teleoperator
# 录制时请按住使能按钮手动移动机械臂
echo "⚠️  重要提示:"
echo "1. 确保机械臂已开机"
echo "2. 录制时请按住使能按钮"
echo "3. 手动拖动机械臂完成演示动作"
echo ""
read -p "按回车键开始录制..." dummy

# 使用自定义录制脚本(因为官方record需要teleop)
python record_rm65_demo.py \
  --left_arm_ip 169.254.128.20 \
  --right_arm_ip 169.254.128.21 \
  --output_dir ~/.cache/huggingface/lerobot/${HF_USER}/${DATASET_NAME} \
  --num_episodes ${NUM_EPISODES} \
  --episode_duration ${EPISODE_TIME_S} \
  --fps ${FPS}

echo ""
echo "============================================================"
echo "录制完成!"
echo "数据集保存在: ~/.cache/huggingface/lerobot/${HF_USER}/${DATASET_NAME}"
echo "============================================================"
