#!/bin/bash

# --- 训练混合数据集（POPQA+ARC+BIO+PUBQA）的T5共识器 (使用GPT-4o-2024-11-20生成的共识) ---

# 1. 训练数据文件 (使用GPT-4o生成的混合数据集)
TRAIN_DATA_FILE="/workspace/conRAG/data/consensus_training/mixed_consensus_for_t5_training_gpt4o_2024_11_20.txt"

# 2. 新模型保存路径 (GPT-4o混合数据集模型)
SAVE_MODEL_PATH="/workspace/conRAG/models/t5_consensus_mixed_gpt4o_2024_11_20"

# 3. 训练参数 (可以根据需要调整)
# 因为数据量变大了，可以适当增加训练轮数(Epochs)或调整学习率
BATCH_SIZE=4       # 如果显存不够，可以调小为 2 或 1
NUM_EPOCHS=6       # 混合数据集2010条，建议多训练几轮避免欠拟合
LEARNING_RATE=3e-5 # 初始学习率
SEED=42

# 4. 指定要使用的GPU (例如，使用第一个GPU: 0)
export CUDA_VISIBLE_DEVICES=1

# --- 执行训练命令 ---
echo "开始训练基于GPT-4o生成的混合数据集的T5共识器..."
echo "训练数据: ${TRAIN_DATA_FILE}"
echo "数据集包含: POPQA(40%), ARC(20%), BIO(20%), PUBQA(20%), 使用GPT-4o-2024-11-20生成"
echo "模型保存路径: ${SAVE_MODEL_PATH}"

# 创建保存模型的目录 (如果不存在)
mkdir -p ${SAVE_MODEL_PATH}

# 调用Python训练脚本
python ../train_consensus.py \
  --train_file "${TRAIN_DATA_FILE}" \
  --save_path "${SAVE_MODEL_PATH}" \
  --batch_size ${BATCH_SIZE} \
  --num_epochs ${NUM_EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --seed ${SEED}

echo "训练完成!"
echo "GPT-4o混合数据集微调模型保存在: ${SAVE_MODEL_PATH}"
echo "请选择loss最低的epoch模型用于后续的共识生成"
echo ""
echo "下一步: 更新增强数据生成脚本中的模型路径"
echo "将 NEW_CONSENSUS_MODEL_DIR 更新为训练好的最佳模型路径"
