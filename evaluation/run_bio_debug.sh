#!/bin/bash
cd /root/shh/FGRAG/fgrag

echo "=== 开始Bio评估调试 ==="
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo "Python版本: $(python --version)"

# 设置详细输出
export PYTHONUNBUFFERED=1

# 添加调试信息
python -u evaluation/evaluate_bio_factscore.py \
    --predictions results/bio/bio_selfrag_enhanced_final_results_0730.jsonl \
    --openai_key sk-eMfsZ6i50RGwa2UFla8EiFuE5UqUAt4rWM363Urmw5AiMfyB \
    --openai_base_url https://yourapi.cn/v1 \
    --num_samples 3 \
    --verbose 2>&1 | tee bio_evaluation_debug.log

echo "=== 评估完成 ==="
echo "日志保存到: bio_evaluation_debug.log"