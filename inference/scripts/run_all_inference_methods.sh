#!/bin/bash
# 统一推理脚本: 为所有数据集运行三种方法 (vanilla, baseline, enhanced) 和两个模型 (llama3:8b, llama3:70b)

# --- 配置 ---
BASE_DIR="/workspace/conRAG"
OLLAMA_BASE_URL="http://172.18.147.77:11434"

# 要测试的模型列表
declare -a models=("llama3:8b" "llama3:70b")
declare -a model_tags=("llama3_8b" "llama3_70b")

# 要运行的方法列表
declare -a methods=("vanilla_rag" "baseline" "enhanced")

# 要处理的数据集列表
declare -a datasets=("popqa" "arc_challenge" "bio" "pubqa")

echo "========================================================================"
echo "开始运行所有推理方法和模型的完整测试"
echo "模型: ${models[@]}"
echo "方法: ${methods[@]}"
echo "数据集: ${datasets[@]}"
echo "========================================================================"

# 循环处理每个模型
for i in "${!models[@]}"; do
    MODEL_NAME="${models[$i]}"
    MODEL_TAG="${model_tags[$i]}"
    
    echo ""
    echo "========================================================================"
    echo "开始处理模型: ${MODEL_NAME} (标签: ${MODEL_TAG})"
    echo "========================================================================"
    
    # 循环处理每个方法
    for METHOD in "${methods[@]}"; do
        echo ""
        echo "------------------------------------------------------------------------"
        echo "运行方法: ${METHOD} | 模型: ${MODEL_NAME}"
        echo "------------------------------------------------------------------------"
        
        # 根据方法选择对应的脚本
        if [ "$METHOD" == "vanilla_rag" ]; then
            SCRIPT_NAME="run_inference_vanilla.sh"
        elif [ "$METHOD" == "baseline" ]; then
            SCRIPT_NAME="run_inference_baseline.sh"
        elif [ "$METHOD" == "enhanced" ]; then
            SCRIPT_NAME="run_inference_enhanced.sh"
        else
            echo "错误: 未知的方法 ${METHOD}"
            continue
        fi
        
        # 临时修改脚本中的模型配置
        TEMP_SCRIPT="/tmp/${SCRIPT_NAME}_${MODEL_TAG}"
        cp "${BASE_DIR}/inference/scripts/${SCRIPT_NAME}" "${TEMP_SCRIPT}"
        
        # 替换模型配置
        sed -i "s/OLLAMA_MODEL_NAME=\".*\"/OLLAMA_MODEL_NAME=\"${MODEL_NAME}\"/" "${TEMP_SCRIPT}"
        sed -i "s/OLLAMA_MODEL_TAG=\".*\"/OLLAMA_MODEL_TAG=\"${MODEL_TAG}\"/" "${TEMP_SCRIPT}"
        
        echo "执行脚本: ${TEMP_SCRIPT}"
        echo "模型配置: ${MODEL_NAME} -> ${MODEL_TAG}"
        
        # 执行推理脚本
        bash "${TEMP_SCRIPT}"
        
        if [ $? -eq 0 ]; then
            echo "✅ ${METHOD} 方法 (${MODEL_NAME}) 执行完成"
        else
            echo "❌ ${METHOD} 方法 (${MODEL_NAME}) 执行失败"
        fi
        
        # 清理临时文件
        rm -f "${TEMP_SCRIPT}"
        
        echo "------------------------------------------------------------------------"
        echo "完成方法: ${METHOD} | 模型: ${MODEL_NAME}"
        echo "------------------------------------------------------------------------"
        
        # 检查生成的结果文件
        echo "检查生成的结果文件..."
        for DATASET in "${datasets[@]}"; do
            if [ "$METHOD" == "vanilla_rag" ]; then
                OUTPUT_FILE="${BASE_DIR}/results/${DATASET}/${DATASET}_vanilla_rag_ollama_${MODEL_TAG}_eval_scientific.jsonl"
            else
                OUTPUT_FILE="${BASE_DIR}/results/${DATASET}/${DATASET}_${METHOD}_ollama_${MODEL_TAG}_eval_scientific.jsonl"
            fi
            
            if [ -f "$OUTPUT_FILE" ]; then
                LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
                echo "✅ ${DATASET}: ${OUTPUT_FILE} (${LINE_COUNT} 行)"
                
                # 显示前几行样例检查格式
                echo "--- ${DATASET} 样例检查 ---"
                head -2 "$OUTPUT_FILE" | jq -r '.query, .answer' 2>/dev/null || head -2 "$OUTPUT_FILE"
                echo "--- 样例检查完毕 ---"
            else
                echo "❌ ${DATASET}: 结果文件未找到 - ${OUTPUT_FILE}"
            fi
        done
        
        # 等待用户确认继续
        echo ""
        read -p "按 Enter 继续下一个方法，或输入 'q' 退出: " user_input
        if [ "$user_input" == "q" ]; then
            echo "用户选择退出"
            exit 0
        fi
    done
    
    echo ""
    echo "========================================================================"
    echo "模型 ${MODEL_NAME} 的所有方法执行完毕"
    echo "========================================================================"
done

echo ""
echo "========================================================================"
echo "🎉 所有推理任务完成！"
echo "========================================================================"
echo "结果文件位于: ${BASE_DIR}/results/"
echo "可以运行评估脚本进行性能评估"
