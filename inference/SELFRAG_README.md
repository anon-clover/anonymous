# ConRAG Self-RAG集成指南

这个集成为ConRAG项目添加了Self-RAG模型支持，通过反思推理来提升问答质量。

## Self-RAG的优势

Self-RAG (Self-Reflective Retrieval-Augmented Generation) 具有以下特殊能力：

1. **反思推理**: 模型会自我评估生成内容的质量和相关性
2. **动态检索决策**: 自动判断何时需要检索外部信息
3. **证据质量评估**: 对检索到的证据进行相关性和支持度评估
4. **自我纠错**: 能够识别和修正错误的推理过程

## 文件结构

```
inference/
├── selfrag_vanilla_rag.py      # Self-RAG Vanilla推理
├── selfrag_baseline_rag.py     # Self-RAG Baseline推理  
├── selfrag_enhanced_rag.py     # Self-RAG Enhanced推理
└── scripts/
    ├── run_selfrag_inference.sh # 批量Self-RAG推理脚本
    └── run_single_selfrag.sh    # 单数据集快速推理脚本
```

## 环境要求

### 必需依赖
```bash
pip install vllm
pip install torch
pip install transformers
pip install tqdm
```

### 模型要求
- Self-RAG模型路径: `/workspace/selfrag/selfrag_llama2_7b`
- 确保模型文件完整且可访问

## 使用方法

### 1. 快速单数据集推理

```bash
cd /workspace/conRAG/inference/scripts

# 基本使用
bash run_single_selfrag.sh pubqa enhanced

# 指定样本数量（用于快速测试）
bash run_single_selfrag.sh pubqa enhanced 50

# 其他示例
bash run_single_selfrag.sh popqa vanilla
bash run_single_selfrag.sh arc_challenge baseline 100
```

### 2. 批量推理所有数据集

```bash
cd /workspace/conRAG/inference/scripts
bash run_selfrag_inference.sh

# 按提示选择数据集和方法
# 或选择 'all' 处理所有数据集和方法
```

### 3. 直接调用Python脚本

```bash
# Self-RAG Enhanced推理示例
python /workspace/conRAG/inference/selfrag_enhanced_rag.py \
    --input_file /workspace/conRAG/data/enhanced_eval/pubqa_eval_scientific_improved_consensus_evidence.jsonl \
    --output_file /workspace/conRAG/results/pubqa/pubqa_selfrag_enhanced_results.jsonl \
    --selfrag_model_path /workspace/selfrag/selfrag_llama2_7b \
    --task pubqa \
    --max_tokens 300 \
    --temperature 0.1 \
    --top_p 0.95
```

## 三种Self-RAG推理方法对比

### Vanilla Self-RAG
- **特点**: 基础的Self-RAG推理
- **输入**: 查询 + 检索文档
- **适用**: 标准RAG场景
- **优势**: 简单直接，反思推理能力强

### Baseline Self-RAG  
- **特点**: 使用所有检索到的文档
- **输入**: 查询 + 多个检索文档
- **适用**: 信息丰富的场景
- **优势**: 信息全面，Self-RAG能评估多文档相关性

### Enhanced Self-RAG
- **特点**: 结合共识分析和额外证据
- **输入**: 查询 + 共识文本 + 额外证据
- **适用**: 复杂推理场景
- **优势**: 信息质量最高，Self-RAG能充分发挥反思能力

## Self-RAG推理特色

### 1. 特殊Prompt格式
```
### Instruction:
{任务指令}

## Input:
{查询和选项}

### Response:
[Retrieval]<paragraph>{检索内容}</paragraph>
```

### 2. 控制Token处理
Self-RAG会生成特殊的控制token：
- `[Fully supported]` / `[Partially supported]` / `[No support / Contradictory]`
- `[Relevant]` / `[Irrelevant]`  
- `[Utility:1-5]`
- `[Continue to Use Evidence]` / `[Retry]`

这些token体现了模型的反思过程，我们会自动清理这些token。

### 3. 任务特定优化

**PubQA**: 针对判断性问题优化
```
"Determine whether the following statement is correct or not. 
Say 'true' if it's correct; otherwise say 'false'. 
Base your judgment on the consensus analysis and additional evidence provided."
```

**ARC Challenge**: 针对多选题优化
```
"Given four answer candidates, A, B, C, and D, choose the best answer choice. 
Use the consensus analysis and additional evidence to carefully evaluate each option."
```

## 推理参数调优

### 建议参数配置

| 参数 | Vanilla | Baseline | Enhanced | 说明 |
|------|---------|----------|----------|------|
| max_tokens | 200 | 250 | 300 | Enhanced需要更多token表达复杂推理 |
| temperature | 0.0 | 0.1 | 0.2 | Enhanced允许更多创造性 |
| top_p | 1.0 | 0.95 | 0.9 | Enhanced使用更多样化的表达 |

### 针对不同任务调优

- **PubQA**: 温度设低(0.0-0.1)，确保判断准确性
- **PopQA**: 温度中等(0.1-0.2)，平衡准确性和流畅性  
- **ARC Challenge**: 温度设低(0.0)，确保选择准确性
- **Bio**: 温度中高(0.2-0.3)，允许更丰富的表达

## 结果分析

### 输出格式
```json
{
    "query": "原始查询",
    "raw_selfrag_response": "Self-RAG模型原始输出(包含控制token)",
    "processed_answer": "清理后的最终答案",
    "method": "selfrag_enhanced",
    "has_consensus": true,
    "num_additional_evidence": 3
}
```

### 分析Self-RAG的反思过程
```bash
# 查看原始Self-RAG输出，观察反思token
cat results/pubqa/pubqa_selfrag_enhanced_results.jsonl | jq -r '.raw_selfrag_response'

# 查看最终处理答案
cat results/pubqa/pubqa_selfrag_enhanced_results.jsonl | jq -r '.processed_answer'

# 统计包含特定反思token的数量
grep -o '\[Fully supported\]' results/pubqa/pubqa_selfrag_enhanced_results.jsonl | wc -l
```

## 性能优化建议

1. **GPU内存**: Self-RAG 7B模型需要约14GB GPU内存
2. **批处理**: VLLM支持高效的批处理推理
3. **模型量化**: 使用`dtype="half"`减少内存使用
4. **长度控制**: 合理设置max_tokens避免过长生成

## 常见问题

### Q: Self-RAG模型加载失败？
A: 确保模型路径正确，且有足够的GPU内存。检查CUDA环境。

### Q: 生成结果包含乱码？
A: 可能是编码问题，确保使用UTF-8编码读写文件。

### Q: 推理速度很慢？
A: 1) 检查GPU使用情况 2) 考虑减少max_tokens 3) 使用VLLM的批处理功能

### Q: 如何观察Self-RAG的反思过程？
A: 查看`raw_selfrag_response`字段中的控制token，它们体现了模型的自我评估过程。

## 与原始推理方法对比

推荐在相同数据上运行原始方法和Self-RAG方法，对比：
1. **答案质量**: Self-RAG的反思能力是否提升了答案准确性
2. **推理过程**: 通过控制token观察Self-RAG的思考过程
3. **错误处理**: Self-RAG是否能更好地识别和避免错误

这种对比能帮助你充分利用Self-RAG的反思推理优势！