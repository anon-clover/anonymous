#!/bin/bash
# 这个脚本使用你新训练的、更强大的T5共识器，为所有数据集生成增强证据 (使用改进的互补性策略)。

# --- 配置 ---
BASE_DIR="/workspace/conRAG"

# --- 重要：更新为你的新T5共识器模型的准确路径 ---
# 使用训练好的混合数据集模型（epoch 6，loss 0.8310）
NEW_CONSENSUS_MODEL_DIR="/workspace/conRAG/models/t5_consensus_mixed_deepseek_2010/best_model_epoch_6_loss_0.8310"

# Sentence Transformer 模型 (保持不变)
ST_MODEL_NAME="/workspace/all-MiniLM-L6-v2"

# --- 要处理的数据集列表 ---
declare -a datasets_to_process=("popqa" "arc_challenge" "bio")

# --- GPU 配置 ---
PROCESSING_GPU_ID=3  # 使用GPU 3，和训练时一样
export CUDA_VISIBLE_DEVICES=${PROCESSING_GPU_ID}
echo "使用 GPU ${CUDA_VISIBLE_DEVICES} 进行证据生成。"

# --- 检查新模型路径是否存在 ---
if [[ "${NEW_CONSENSUS_MODEL_DIR}" == *"/best_model_epoch_4_loss_0.XXXX"* ]]; then
    echo "错误：请在脚本中将 NEW_CONSENSUS_MODEL_DIR 更新为你的实际最佳模型路径！"
    exit 1
fi
if [ ! -d "${NEW_CONSENSUS_MODEL_DIR}" ]; then
    echo "错误: 新的T5共识器模型目录未找到: ${NEW_CONSENSUS_MODEL_DIR}"
    exit 1
fi

# --- 循环处理每个数据集 ---
for DATASET_NAME in "${datasets_to_process[@]}"; do
    echo ""
    echo "========================================================================"
    echo "--- 开始为数据集: ${DATASET_NAME} 生成增强证据 ---"

    INPUT_RETRIEVED_FILE="${BASE_DIR}/data/${DATASET_NAME}/${DATASET_NAME}_retrieved.jsonl"
    OUTPUT_ENHANCED_FILE="${BASE_DIR}/data/${DATASET_NAME}/${DATASET_NAME}_enhanced_consensus_evidence_deepseek_t5.jsonl" # 用新名字以区分

    if [ ! -f "${INPUT_RETRIEVED_FILE}" ]; then
        echo "警告: ${DATASET_NAME} 的输入文件未找到，已跳过: ${INPUT_RETRIEVED_FILE}"
        continue
    fi

    echo "正在使用新模型为 ${DATASET_NAME} 生成证据 (任务类型: ${TASK_TYPE})..."
    echo "输入: ${INPUT_RETRIEVED_FILE}"
    echo "输出: ${OUTPUT_ENHANCED_FILE}"
    echo "策略: 互补性证据提取 (改进版)"

    # 提取任务类型
    TASK_TYPE=$(echo "${DATASET_NAME}" | sed 's/_retrieved//')

    python "${BASE_DIR}/data_processing/generate_enhanced_consensus.py" \
      --input_file "${INPUT_RETRIEVED_FILE}" \
      --output_file "${OUTPUT_ENHANCED_FILE}" \
      --consensus_model_path "${NEW_CONSENSUS_MODEL_DIR}" \
      --st_model_name "${ST_MODEL_NAME}" \
      --device "cuda" \
      --task_type "${TASK_TYPE}" \
      --max_evidence 3 \
      --min_evidence 1

    if [ $? -ne 0 ]; then
        echo "错误: 为 ${DATASET_NAME} 生成证据失败。"
    else
        echo "成功为 ${DATASET_NAME} 生成证据！"
    fi
    echo "========================================================================"
done

echo "--- 所有数据集的增强证据已生成完毕 ---"