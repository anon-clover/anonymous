#!/bin/bash
# 自动化执行llama3:70b的所有推理任务
# 按顺序执行：Vanilla → Baseline → Enhanced

BASE_DIR="/workspace/conRAG"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "========================================================================"
echo "开始执行 llama3:70b 的所有推理任务"
echo "执行顺序: Vanilla RAG → Baseline RAG → Enhanced RAG"
echo "预计总时间: 6-8小时 (70b模型较慢)"
echo "========================================================================"

# 记录开始时间
START_TIME=$(date)
echo "开始时间: ${START_TIME}"

# 1. 执行 Vanilla RAG
echo ""
echo "------------------------------------------------------------------------"
echo "第1步: 执行 Vanilla RAG (llama3:70b)"
echo "------------------------------------------------------------------------"
cd "${BASE_DIR}/inference/scripts"
bash run_inference_vanilla.sh > "${LOG_DIR}/vanilla_70b.log" 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Vanilla RAG (70b) 执行完成"
    echo "完成时间: $(date)"
else
    echo "❌ Vanilla RAG (70b) 执行失败"
    echo "请检查日志: ${LOG_DIR}/vanilla_70b.log"
    exit 1
fi

# 2. 执行 Baseline RAG  
echo ""
echo "------------------------------------------------------------------------"
echo "第2步: 执行 Baseline RAG (llama3:70b)"
echo "------------------------------------------------------------------------"
bash run_inference_baseline.sh > "${LOG_DIR}/baseline_70b.log" 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Baseline RAG (70b) 执行完成"
    echo "完成时间: $(date)"
else
    echo "❌ Baseline RAG (70b) 执行失败"
    echo "请检查日志: ${LOG_DIR}/baseline_70b.log"
    exit 1
fi

# 3. 执行 Enhanced RAG
echo ""
echo "------------------------------------------------------------------------"
echo "第3步: 执行 Enhanced RAG (llama3:70b)"
echo "------------------------------------------------------------------------"
bash run_inference_enhanced.sh > "${LOG_DIR}/enhanced_70b.log" 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Enhanced RAG (70b) 执行完成"
    echo "完成时间: $(date)"
else
    echo "❌ Enhanced RAG (70b) 执行失败"
    echo "请检查日志: ${LOG_DIR}/enhanced_70b.log"
    exit 1
fi

# 记录结束时间
END_TIME=$(date)
echo ""
echo "========================================================================"
echo "🎉 所有 llama3:70b 推理任务完成！"
echo "========================================================================"
echo "开始时间: ${START_TIME}"
echo "结束时间: ${END_TIME}"

# 统计结果文件
echo ""
echo "结果文件统计:"
wc -l "${BASE_DIR}/results/*/*_*_ollama_llama3_70b_eval_scientific.jsonl"

echo ""
echo "日志文件位置:"
echo "- Vanilla RAG: ${LOG_DIR}/vanilla_70b.log"
echo "- Baseline RAG: ${LOG_DIR}/baseline_70b.log"
echo "- Enhanced RAG: ${LOG_DIR}/enhanced_70b.log"

echo ""
echo "所有推理任务已完成，可以开始评估！"
