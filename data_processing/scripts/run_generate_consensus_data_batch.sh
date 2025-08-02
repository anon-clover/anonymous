#!/bin/bash
# 批量为所有数据集生成共识训练数据 (使用GPT-4o-2024-11-20)

# --- 配置 ---
BASE_DIR="/workspace/conRAG"
NUM_SAMPLES=500  # 每个数据集生成的样本数量

# --- 要处理的数据集列表 ---
declare -a datasets=("popqa" "arc_challenge" "bio" "pubqa")

echo "========================================================================"
echo "批量生成共识训练数据 (使用GPT-4o-2024-11-20)"
echo "========================================================================"
echo "基础目录: ${BASE_DIR}"
echo "每个数据集样本数: ${NUM_SAMPLES}"
echo "处理数据集: ${datasets[*]}"
echo ""

# --- 循环处理每个数据集 ---
for dataset in "${datasets[@]}"; do
    echo "----------------------------------------"
    echo "处理数据集: ${dataset}"
    echo "----------------------------------------"
    
    # 输入和输出文件路径
    INPUT_FILE="${BASE_DIR}/data/${dataset}/${dataset}_retrieved.jsonl"
    OUTPUT_FILE="${BASE_DIR}/data/${dataset}/${dataset}_consensus_for_t5_training_gpt4o_2024_11_20_${NUM_SAMPLES}_samples.txt"
    
    # 检查输入文件是否存在
    if [ ! -f "$INPUT_FILE" ]; then
        echo "警告: 输入文件不存在: $INPUT_FILE"
        echo "跳过数据集: $dataset"
        continue
    fi
    
    echo "输入文件: $INPUT_FILE"
    echo "输出文件: $OUTPUT_FILE"
    
    # 创建临时的Python脚本来处理当前数据集
    TEMP_SCRIPT="${BASE_DIR}/data_processing/temp_generate_consensus_${dataset}.py"
    
    cat > "$TEMP_SCRIPT" << EOF
import os
import sys
sys.path.append('${BASE_DIR}')

# 修改openai_generate_consensus_data.py中的路径配置
from data_processing.openai_generate_consensus_data import *

def main():
    # 为当前数据集配置路径
    input_file_path = '${INPUT_FILE}'
    output_file_path = '${OUTPUT_FILE}'
    num_records_to_process = ${NUM_SAMPLES}

    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    print(f"Reading data from: {input_file_path}")
    all_data = read_jsonl_data(input_file_path)

    if not all_data:
        print(f"No data successfully read from {input_file_path}")
        return

    # 选择要处理的数据子集
    data_to_process = all_data[:num_records_to_process]
    actual_records_to_process = len(data_to_process)

    if actual_records_to_process == 0:
        print("No records to process")
        return

    print(f"Found {len(all_data)} total items. Will process the first {actual_records_to_process} items.")

    # 清空或创建输出文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("")
    print(f"Output will be saved to: {output_file_path}")

    for item in tqdm(data_to_process, desc=f"Generating Consensus for {actual_records_to_process} items with GPT-4o-2024-11-20"):
        query = item["query"]
        documents = item["passages"]

        consensus_text = generate_consensus_with_openai(query, documents)

        if consensus_text:
            write_consensus_to_file(output_file_path, query, documents, consensus_text)
        else:
            print(f"Failed to generate consensus for query: {query}")
            write_consensus_to_file(output_file_path, query, documents, "CONSENSUS_GENERATION_FAILED")

    print(f"\\nConsensus generation complete for {actual_records_to_process} items. Output saved to: {output_file_path}")

if __name__ == '__main__':
    main()
EOF

    # 运行临时脚本
    echo "开始生成共识数据..."
    python "$TEMP_SCRIPT"
    
    if [ $? -eq 0 ]; then
        echo "✅ 数据集 $dataset 处理完成"
        
        # 显示生成的数据统计
        if [ -f "$OUTPUT_FILE" ]; then
            line_count=$(wc -l < "$OUTPUT_FILE")
            echo "生成的训练样本数: $line_count"
        fi
    else
        echo "❌ 数据集 $dataset 处理失败"
    fi
    
    # 清理临时脚本
    rm -f "$TEMP_SCRIPT"
    
    echo ""
done

echo "========================================================================"
echo "批量共识数据生成完成"
echo "========================================================================"
echo ""
echo "生成的文件:"
for dataset in "${datasets[@]}"; do
    OUTPUT_FILE="${BASE_DIR}/data/${dataset}/${dataset}_consensus_for_t5_training_gpt4o_2024_11_20_${NUM_SAMPLES}_samples.txt"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✅ $OUTPUT_FILE"
    else
        echo "❌ $OUTPUT_FILE (未生成)"
    fi
done

echo ""
echo "下一步: 使用生成的共识数据训练T5模型"
echo "1. 将所有生成的文件合并为混合训练数据"
echo "2. 运行 training/scripts/run_train_consensus.sh"
