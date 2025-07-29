#!/bin/bash

# ConRAG Self-RAG推理脚本
# 支持三种推理方法：vanilla, baseline, enhanced

set -e

echo "=== ConRAG Self-RAG推理系统 ==="

# 配置路径
BASE_DIR="/workspace/conRAG"
SELFRAG_MODEL_PATH="/workspace/selfrag/selfrag_llama2_7b"
DATA_DIR="$BASE_DIR/data/enhanced_eval"
RESULTS_DIR="$BASE_DIR/results"

# 检查Self-RAG模型是否存在
if [ ! -d "$SELFRAG_MODEL_PATH" ]; then
    echo "错误: Self-RAG模型未找到: $SELFRAG_MODEL_PATH"
    echo "请确保Self-RAG模型已正确安装到指定路径"
    exit 1
fi

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 数据集配置
declare -a datasets=("popqa" "arc_challenge" "bio" "pubqa")
declare -a methods=("vanilla" "baseline" "enhanced")

# 用户选择
echo "可用数据集: ${datasets[*]}"
read -p "选择数据集 (或输入 'all' 处理所有数据集): " selected_dataset

echo "可用方法: ${methods[*]}"
read -p "选择推理方法 (或输入 'all' 使用所有方法): " selected_method

# Self-RAG推理参数
MAX_TOKENS=300
TEMPERATURE=0.1
TOP_P=0.95
DEVICE="cuda:1"
NUM_SAMPLES=-1  # -1表示处理全部样本

echo "Self-RAG推理参数:"
echo "- 最大token数: $MAX_TOKENS"
echo "- 温度: $TEMPERATURE"  
echo "- Top-p: $TOP_P"
echo "- 设备: $DEVICE"
echo "- 样本数量: $NUM_SAMPLES (-1表示全部)"

read -p "是否修改默认参数? (y/n): " modify_params
if [[ $modify_params == "y" || $modify_params == "Y" ]]; then
    read -p "最大token数 [$MAX_TOKENS]: " input_tokens
    if [[ -n "$input_tokens" ]]; then
        MAX_TOKENS=$input_tokens
    fi
    
    read -p "温度 [$TEMPERATURE]: " input_temp
    if [[ -n "$input_temp" ]]; then
        TEMPERATURE=$input_temp
    fi
    
    read -p "Top-p [$TOP_P]: " input_top_p
    if [[ -n "$input_top_p" ]]; then
        TOP_P=$input_top_p
    fi
    
    read -p "样本数量 [$NUM_SAMPLES]: " input_samples
    if [[ -n "$input_samples" ]]; then
        NUM_SAMPLES=$input_samples
    fi
fi

# 处理数据集选择
if [[ "$selected_dataset" == "all" ]]; then
    datasets_to_process=("${datasets[@]}")
else
    datasets_to_process=("$selected_dataset")
fi

# 处理方法选择
if [[ "$selected_method" == "all" ]]; then
    methods_to_process=("${methods[@]}")
else
    methods_to_process=("$selected_method")
fi

echo "=== 开始Self-RAG推理 ==="
echo "数据集: ${datasets_to_process[*]}"
echo "方法: ${methods_to_process[*]}"
echo "Self-RAG模型: $SELFRAG_MODEL_PATH"

# 执行推理
for dataset in "${datasets_to_process[@]}"; do
    for method in "${methods_to_process[@]}"; do
        echo ""
        echo "========================================================================"
        echo "正在处理: $dataset 数据集，使用 Self-RAG $method 方法"
        echo "========================================================================"
        
        # 确定输入文件
        if [[ "$method" == "enhanced" ]]; then
            # Enhanced方法使用改进后的数据（如果存在）
            if [[ "$dataset" == "pubqa" ]] && [ -f "$DATA_DIR/${dataset}_eval_scientific_improved_consensus_evidence.jsonl" ]; then
                INPUT_FILE="$DATA_DIR/${dataset}_eval_scientific_improved_consensus_evidence.jsonl"
            else
                INPUT_FILE="$DATA_DIR/${dataset}_eval_scientific_enhanced_consensus_evidence.jsonl"
            fi
        else
            # Vanilla和Baseline方法使用标准增强数据
            INPUT_FILE="$DATA_DIR/${dataset}_eval_scientific_enhanced_consensus_evidence.jsonl"
        fi
        
        # 检查输入文件是否存在
        if [ ! -f "$INPUT_FILE" ]; then
            echo "警告: 输入文件不存在，跳过: $INPUT_FILE"
            continue
        fi
        
        # 创建数据集结果目录
        mkdir -p "$RESULTS_DIR/$dataset"
        
        # 确定输出文件
        OUTPUT_FILE="$RESULTS_DIR/$dataset/${dataset}_selfrag_${method}_results.jsonl"
        
        # 确定使用的脚本
        SCRIPT_PATH="$BASE_DIR/inference/selfrag_${method}_rag.py"
        
        if [ ! -f "$SCRIPT_PATH" ]; then
            echo "错误: 脚本文件不存在: $SCRIPT_PATH"
            continue
        fi
        
        echo "输入文件: $INPUT_FILE"
        echo "输出文件: $OUTPUT_FILE"
        echo "推理脚本: $SCRIPT_PATH"
        
        # 执行Self-RAG推理
        python "$SCRIPT_PATH" \
            --input_file "$INPUT_FILE" \
            --output_file "$OUTPUT_FILE" \
            --selfrag_model_path "$SELFRAG_MODEL_PATH" \
            --task "$dataset" \
            --max_tokens "$MAX_TOKENS" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --device "$DEVICE" \
            --num_samples "$NUM_SAMPLES"
        
        if [ $? -eq 0 ]; then
            echo "✅ Self-RAG $method 推理完成: $dataset"
            echo "结果保存到: $OUTPUT_FILE"
        else
            echo "❌ Self-RAG $method 推理失败: $dataset"
        fi
    done
done

echo ""
echo "=== Self-RAG推理完成 ==="
echo "所有结果保存在: $RESULTS_DIR"
echo ""
echo "可以使用以下命令查看结果:"
echo "ls -la $RESULTS_DIR/*/"
echo ""
echo "推荐下一步:"
echo "1. 运行评估脚本分析Self-RAG的性能"
echo "2. 对比不同方法的推理结果"
echo "3. 分析Self-RAG的反思推理能力"