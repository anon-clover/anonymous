#!/bin/bash
# 这个脚本使用训练好的T5共识器，为评估数据集生成增强证据 (使用改进的互补性策略)

# --- 配置 ---
BASE_DIR="/workspace/conRAG"

# --- T5共识器模型路径 ---
# 使用GPT-4o混合数据集训练的最佳模型 (epoch 6, loss 0.6528)
NEW_CONSENSUS_MODEL_DIR="/workspace/conRAG/models/t5_consensus_mixed_gpt4o_2024_11_20/best_model_epoch_6_loss_0.6528"

# Sentence Transformer 模型
ST_MODEL_NAME="/workspace/all-MiniLM-L6-v2"

# --- GPU 配置 ---
PROCESSING_GPU_ID=3
export CUDA_VISIBLE_DEVICES=${PROCESSING_GPU_ID}
echo "使用 GPU ${CUDA_VISIBLE_DEVICES} 进行证据生成。"

# --- 检查模型路径是否存在 ---
if [ ! -d "${NEW_CONSENSUS_MODEL_DIR}" ]; then
    echo "错误: T5共识器模型目录未找到: ${NEW_CONSENSUS_MODEL_DIR}"
    exit 1
fi

echo "========================================================================"
echo "注意：此脚本仅为评估数据集生成增强证据"
echo "训练数据集不会被处理，以避免数据泄露"
echo "========================================================================"

# --- 使用科学分割的测试集（这是正确的选择）---
declare -a eval_datasets=(
    "${BASE_DIR}/data/popqa/splits/popqa_eval_scientific.jsonl"
    "${BASE_DIR}/data/arc_challenge/splits/arc_challenge_eval_scientific.jsonl"
    "${BASE_DIR}/data/bio/splits/bio_eval_scientific.jsonl"
    "${BASE_DIR}/data/pubqa/splits/pubqa_eval_scientific.jsonl"
)

# 注意：如果splits文件夹不存在，需要先运行科学分割脚本：
# cd ${BASE_DIR}/data_processing/scripts
# bash run_scientific_split.sh

# --- 循环处理每个评估数据集 ---
for INPUT_FILE in "${eval_datasets[@]}"; do
    if [ ! -f "${INPUT_FILE}" ]; then
        echo "警告: 评估文件未找到，已跳过: ${INPUT_FILE}"
        continue
    fi
    
    # 从路径中提取数据集名称
    FILENAME=$(basename "${INPUT_FILE}")
    DATASET_NAME="${FILENAME%.*}"

    # 提取任务类型 (popqa, arc_challenge, bio, pubqa)
    TASK_TYPE=$(echo "${DATASET_NAME}" | sed 's/_eval_scientific//')

    # 输出文件路径
    OUTPUT_DIR="${BASE_DIR}/data/enhanced_eval"
    mkdir -p "${OUTPUT_DIR}"
    OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_enhanced_consensus_evidence.jsonl"
    
    echo ""
    echo "========================================================================"
    echo "--- 开始为评估数据集生成增强证据: ${DATASET_NAME} (任务类型: ${TASK_TYPE}) ---"
    echo "输入: ${INPUT_FILE}"
    echo "输出: ${OUTPUT_FILE}"
    echo "策略: 互补性证据提取 (改进版)"

    python "${BASE_DIR}/data_processing/generate_enhanced_consensus.py" \
      --input_file "${INPUT_FILE}" \
      --output_file "${OUTPUT_FILE}" \
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

echo ""
echo "--- 所有评估数据集的增强证据已生成完毕 ---"
echo "输出文件位于: ${BASE_DIR}/data/enhanced_eval/"
echo ""
echo "接下来可以运行推理脚本进行评估："
echo "cd ${BASE_DIR}/inference/scripts"
echo "bash run_inference_enhanced.sh"