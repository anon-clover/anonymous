#!/bin/bash

# 单数据集Self-RAG推理脚本 - 用于快速测试

# 使用方法:
# bash run_single_selfrag.sh <dataset> <method> [num_samples]
# 示例: bash run_single_selfrag.sh pubqa enhanced 50

set -e

# 参数检查
if [ $# -lt 2 ]; then
    echo "使用方法: bash run_single_selfrag.sh <dataset> <method> [num_samples]"
    echo "数据集: popqa, arc_challenge, bio, pubqa"
    echo "方法: vanilla, baseline, enhanced"  
    echo "样本数量: 可选，默认处理全部"
    echo ""
    echo "示例:"
    echo "  bash run_single_selfrag.sh pubqa enhanced 50"
    echo "  bash run_single_selfrag.sh popqa vanilla"
    exit 1
fi

DATASET=$1
METHOD=$2
NUM_SAMPLES=${3:--1}  # 默认-1表示处理全部

# 配置路径
BASE_DIR="/workspace/conRAG"
SELFRAG_MODEL_PATH="/workspace/selfrag/selfrag_llama2_7b"
DATA_DIR="$BASE_DIR/data/enhanced_eval"
RESULTS_DIR="$BASE_DIR/results"

echo "=== Self-RAG单数据集快速推理 ==="
echo "数据集: $DATASET"
echo "方法: Self-RAG $METHOD"
echo "样本数量: $NUM_SAMPLES"
echo "Self-RAG模型: $SELFRAG_MODEL_PATH"

# 检查Self-RAG模型
if [ ! -d "$SELFRAG_MODEL_PATH" ]; then
    echo "错误: Self-RAG模型未找到: $SELFRAG_MODEL_PATH"
    exit 1
fi

# 确定输入文件
if [[ "$METHOD" == "enhanced" && "$DATASET" == "pubqa" ]] && [ -f "$DATA_DIR/pubqa_eval_scientific_improved_consensus_evidence.jsonl" ]; then
    INPUT_FILE="$DATA_DIR/pubqa_eval_scientific_improved_consensus_evidence.jsonl"
    echo "使用改进的PubQA数据"
else
    INPUT_FILE="$DATA_DIR/${DATASET}_eval_scientific_enhanced_consensus_evidence.jsonl"
fi

# 检查输入文件
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$RESULTS_DIR/$DATASET"
OUTPUT_FILE="$RESULTS_DIR/$DATASET/${DATASET}_selfrag_${METHOD}_results.jsonl"

# 确定推理脚本
SCRIPT_PATH="$BASE_DIR/inference/selfrag_${METHOD}_rag.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 推理脚本不存在: $SCRIPT_PATH"
    exit 1
fi

# Self-RAG推理参数
MAX_TOKENS=300
TEMPERATURE=0.1
TOP_P=0.95
DEVICE="cuda:1"

echo ""
echo "推理配置:"
echo "- 输入文件: $INPUT_FILE"
echo "- 输出文件: $OUTPUT_FILE"
echo "- 推理脚本: $SCRIPT_PATH"
echo "- 最大token: $MAX_TOKENS"
echo "- 温度: $TEMPERATURE"
echo "- Top-p: $TOP_P"
echo "- 设备: $DEVICE"
echo ""

# 执行Self-RAG推理
echo "开始Self-RAG推理..."
python "$SCRIPT_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --selfrag_model_path "$SELFRAG_MODEL_PATH" \
    --task "$DATASET" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --device "$DEVICE" \
    --num_samples "$NUM_SAMPLES"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Self-RAG推理成功完成!"
    echo "结果保存到: $OUTPUT_FILE"
    
    # 显示结果统计
    if [ -f "$OUTPUT_FILE" ]; then
        TOTAL_LINES=$(wc -l < "$OUTPUT_FILE")
        echo "生成结果数量: $TOTAL_LINES"
        
        # 显示前几个结果示例
        echo ""
        echo "=== 结果示例 ==="
        head -2 "$OUTPUT_FILE" | jq -r '.processed_answer' 2>/dev/null || head -2 "$OUTPUT_FILE"
    fi
    
    echo ""
    echo "可以使用以下命令查看完整结果:"
    echo "cat $OUTPUT_FILE | jq '.processed_answer'"
    echo ""
    echo "或查看原始Self-RAG输出:"
    echo "cat $OUTPUT_FILE | jq '.raw_selfrag_response'"
    
else
    echo "❌ Self-RAG推理失败"
    exit 1
fi