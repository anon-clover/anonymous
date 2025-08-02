#!/bin/bash
# 科学数据划分脚本

# --- 配置 ---
BASE_DIR="/workspace/conRAG"
TRAIN_RATIO=0.5
SEED=42

echo "=== 执行科学数据划分 ==="
echo "基础目录: ${BASE_DIR}"
echo "训练比例: ${TRAIN_RATIO}"
echo "随机种子: ${SEED}"

# 执行数据划分
python "${BASE_DIR}/data_processing/scientific_data_split.py" \
    --base_dir "${BASE_DIR}" \
    --train_ratio ${TRAIN_RATIO} \
    --seed ${SEED} \
    --datasets popqa arc_challenge bio pubqa \
    --experiment_type both

echo ""
echo "=== 划分后的数据结构 ==="
echo "data/"
echo "├── popqa/splits/"
echo "│   ├── popqa_train_scientific.jsonl     (科学训练集)"
echo "│   ├── popqa_eval_scientific.jsonl      (科学测试集)"
echo "│   └── popqa_full_for_application.jsonl (应用评估集)"
echo "├── arc_challenge/splits/"
echo "│   ├── arc_challenge_train_scientific.jsonl"
echo "│   ├── arc_challenge_eval_scientific.jsonl"
echo "│   └── arc_challenge_full_for_application.jsonl"
echo "├── bio/splits/"
echo "│   ├── bio_train_scientific.jsonl"
echo "│   ├── bio_eval_scientific.jsonl"
echo "│   └── bio_full_for_application.jsonl"
echo "├── pubqa/splits/"
echo "│   ├── pubqa_train_scientific.jsonl"
echo "│   ├── pubqa_eval_scientific.jsonl"
echo "│   └── pubqa_full_for_application.jsonl"
echo "└── consensus_training/"
echo "    └── mixed_consensus_training_data.jsonl (混合共识器训练数据)"
echo ""
echo "=== 实验建议 ==="
echo "第一步: 用 mixed_consensus_training_data.jsonl 训练共识器"
echo "第二步: 在 *_eval_scientific.jsonl 上做严格科学验证"
echo "第三步: 在 *_full_for_application.jsonl 上展示应用效果"
echo "现在使用5:5划分，有足够的评估数据！"