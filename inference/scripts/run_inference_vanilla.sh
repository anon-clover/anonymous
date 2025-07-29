#!/bin/bash
# 纯RAG推理脚本: 为所有数据集运行 Vanilla RAG 推理 (不使用共识，不使用额外证据)

# --- 配置 ---
BASE_DIR="/workspace/conRAG"
OLLAMA_BASE_URL="http://172.18.147.77:11434" # 你的Ollama API地址
OLLAMA_MODEL_NAME="llama3:8b"
OLLAMA_MODEL_TAG="llama3_8b"

# --- 要运行推理的数据集列表 ---
declare -a datasets_to_process=("popqa" "arc_challenge" "bio" "pubqa")

############################################################
# 第一部分: 准备评估数据
############################################################
echo "--- 准备评估数据 ---"

# 1. 首先提取评估集对应的retrieved数据
echo "提取评估集对应的retrieved数据..."
python "${BASE_DIR}/data_processing/extract_eval_retrieved.py" \
    --base_dir "${BASE_DIR}" \
    --datasets "popqa" "arc_challenge" "bio"

# 2. 准备ARC Challenge数据（合并选项）
echo "--- 准备 ARC Challenge 评估数据：合并选项和答案 ---"
ARC_RETRIEVED_FILE="${BASE_DIR}/data/arc_challenge/splits/arc_challenge_eval_retrieved.jsonl"
ARC_ORIGINAL_FILE_WITH_CHOICES="${BASE_DIR}/eval_data/arc_challenge_processed.jsonl"
ARC_VANILLA_INPUT_FILE="${BASE_DIR}/data/arc_challenge/arc_challenge_vanilla_eval_input.jsonl"

# 检查所需文件是否存在
if [ -f "${ARC_RETRIEVED_FILE}" ] && [ -f "${ARC_ORIGINAL_FILE_WITH_CHOICES}" ]; then
    python "${BASE_DIR}/data_processing/merge_arc_choices.py" \
        --enhanced_file "${ARC_RETRIEVED_FILE}" \
        --original_file_with_choices "${ARC_ORIGINAL_FILE_WITH_CHOICES}" \
        --output_file "${ARC_VANILLA_INPUT_FILE}"
    echo "--- ARC Challenge 数据准备完毕 ---"
else
    echo "警告: 准备 ARC Challenge 数据所需的文件不完整，将跳过合并步骤。"
fi

############################################################
# 第二部分: 运行纯RAG推理
############################################################
echo "--- 开始为所有数据集运行 Vanilla RAG 推理 (LLM: ${OLLAMA_MODEL_NAME}) ---"
echo "注意: 这是最基础的RAG方法，只使用检索到的文档，不进行共识处理，不生成额外证据"

for DATASET_NAME in "${datasets_to_process[@]}"; do
    echo ""
    echo "------------------------------------------------------------------------"
    echo "运行 Vanilla RAG 推理: 数据集=${DATASET_NAME}"
    echo "------------------------------------------------------------------------"

    # 根据数据集选择正确的输入文件
    INPUT_FILE=""
    if [ "$DATASET_NAME" == "arc_challenge" ]; then
        INPUT_FILE="${ARC_VANILLA_INPUT_FILE}"
    else
        # 使用增强后的评估文件，因为vanilla方法只使用其中的原始段落
        INPUT_FILE="${BASE_DIR}/data/enhanced_eval/${DATASET_NAME}_eval_scientific_enhanced_consensus_evidence.jsonl"
    fi
    
    OUTPUT_JSONL="${BASE_DIR}/results/${DATASET_NAME}/${DATASET_NAME}_vanilla_rag_ollama_${OLLAMA_MODEL_TAG}_eval_scientific.jsonl"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "警告: 输入文件 ${INPUT_FILE} 未找到，跳过 ${DATASET_NAME} 的推理..."
        continue
    fi

    echo "输入: ${INPUT_FILE}"
    echo "输出: ${OUTPUT_JSONL}"

    # 创建输出目录
    mkdir -p "$(dirname "$OUTPUT_JSONL")"

    # 运行纯RAG推理
    python "${BASE_DIR}/inference/vanilla_rag.py" \
      --input_file "${INPUT_FILE}" \
      --output_file "${OUTPUT_JSONL}" \
      --ollama_base_url "${OLLAMA_BASE_URL}" \
      --ollama_model_name "${OLLAMA_MODEL_NAME}" \
      --task "${DATASET_NAME}"
    
    if [ $? -eq 0 ]; then
        echo "数据集 ${DATASET_NAME} 的 Vanilla RAG 推理完成。"
    else
        echo "错误: 数据集 ${DATASET_NAME} 的 Vanilla RAG 推理失败。"
    fi
done

echo ""
echo "--- 所有数据集的 Vanilla RAG 推理任务已全部完成 ---"
echo ""
echo "=== 方法对比说明 ==="
echo "1. Vanilla RAG: 只使用检索到的原始文档"
echo "2. Baseline RAG: 使用共识处理的文档"  
echo "3. Enhanced RAG: 使用共识+额外证据的文档"
echo "==================="