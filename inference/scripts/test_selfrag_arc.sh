#!/bin/bash

# SelfRAG ARC Challenge测试脚本
# 使用方法: bash test_selfrag_arc.sh

echo "开始测试SelfRAG ARC Challenge..."

# 设置基本路径
PROJECT_ROOT="/workspace/conRAG"
INPUT_FILE="${PROJECT_ROOT}/data/enhanced_eval/arc_challenge_eval_scientific_enhanced_consensus_evidence.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/results/arc"
SCRIPT_PATH="${PROJECT_ROOT}/inference/selfrag_enhanced_rag.py"

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 测试参数
NUM_SAMPLES=50
MAX_TOKENS=50
TEMPERATURE=0.0
DEVICE="cuda:1"

echo "测试参数:"
echo "  - 样本数量: $NUM_SAMPLES"
echo "  - 最大tokens: $MAX_TOKENS" 
echo "  - 温度: $TEMPERATURE"
echo "  - 设备: $DEVICE"
echo ""

# 执行测试
echo "正在执行SelfRAG Enhanced测试..."
python $SCRIPT_PATH \
    --input_file $INPUT_FILE \
    --output_file "${OUTPUT_DIR}/arc_selfrag_enhanced_test.jsonl" \
    --task arc_challenge \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --device $DEVICE \
    --num_samples $NUM_SAMPLES

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "测试完成！正在检查结果格式..."
    
    # 检查输出文件是否存在
    if [ -f "${OUTPUT_DIR}/arc_selfrag_enhanced_test.jsonl" ]; then
        echo "输出文件已生成: ${OUTPUT_DIR}/arc_selfrag_enhanced_test.jsonl"
        
        # 显示前10个答案
        echo ""
        echo "前10个处理后的答案:"
        head -10 "${OUTPUT_DIR}/arc_selfrag_enhanced_test.jsonl" | jq -r '.processed_answer' | nl
        
        # 统计答案分布
        echo ""
        echo "答案分布统计:"
        jq -r '.processed_answer' "${OUTPUT_DIR}/arc_selfrag_enhanced_test.jsonl" | sort | uniq -c | sort -nr
        
        # 检查是否都是ABCD格式
        non_abcd_count=$(jq -r '.processed_answer' "${OUTPUT_DIR}/arc_selfrag_enhanced_test.jsonl" | grep -v '^[ABCD]$' | wc -l)
        total_count=$(wc -l < "${OUTPUT_DIR}/arc_selfrag_enhanced_test.jsonl")
        
        echo ""
        echo "格式检查结果:"
        echo "  - 总样本数: $total_count"
        echo "  - 非ABCD格式: $non_abcd_count"
        echo "  - ABCD格式率: $(( (total_count - non_abcd_count) * 100 / total_count ))%"
        
        if [ $non_abcd_count -eq 0 ]; then
            echo "✅ 所有答案都是正确的ABCD格式！"
            echo ""
            echo "如果要运行完整数据集，执行:"
            echo "bash run_full_arc.sh"
        else
            echo "⚠️  发现非ABCD格式的答案，需要进一步优化"
            echo ""
            echo "非ABCD格式的答案示例:"
            jq -r '.processed_answer' "${OUTPUT_DIR}/arc_selfrag_enhanced_test.jsonl" | grep -v '^[ABCD]$' | head -5
        fi
    else
        echo "❌ 输出文件未生成，请检查错误信息"
    fi
else
    echo "❌ 测试执行失败，请检查错误信息"
fi

echo ""
echo "测试脚本执行完毕"