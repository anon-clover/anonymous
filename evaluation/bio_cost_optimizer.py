#!/usr/bin/env python3
"""
BIO评估成本优化策略
解决FactScore评估的高成本问题
"""

import argparse
import json
import os
from tqdm import tqdm
import random

def estimate_bio_evaluation_cost(num_samples, avg_facts_per_sample=5):
    """估算BIO评估成本"""
    
    # 每个样本的API调用次数
    calls_per_sample = 1 + avg_facts_per_sample  # 1次提取事实 + N次验证事实
    total_calls = num_samples * calls_per_sample
    
    # 估算token消耗
    extract_tokens_per_call = 1000  # generate_atomic_facts max_tokens
    verify_tokens_per_call = 10     # verify_atomic_fact max_tokens
    
    total_extract_tokens = num_samples * extract_tokens_per_call
    total_verify_tokens = num_samples * avg_facts_per_sample * verify_tokens_per_call
    total_tokens = total_extract_tokens + total_verify_tokens
    
    # 估算成本 (按GPT-3.5-turbo计算: $0.002/1K tokens)
    estimated_cost_usd = total_tokens * 0.002 / 1000
    
    print(f"📊 BIO评估成本估算")
    print(f"样本数量: {num_samples}")
    print(f"平均事实数/样本: {avg_facts_per_sample}")
    print(f"总API调用次数: {total_calls}")
    print(f"总token消耗: {total_tokens:,}")
    print(f"估算成本: ${estimated_cost_usd:.2f} USD")
    print(f"每样本成本: ${estimated_cost_usd/num_samples:.4f} USD")
    
    return total_calls, total_tokens, estimated_cost_usd

def create_bio_evaluation_strategy():
    """创建BIO评估策略"""
    
    print("🎯 BIO评估策略建议")
    print("="*50)
    
    # 策略1: 分阶段评估
    print("\n📋 策略1: 分阶段评估")
    print("阶段1: 小样本测试 (10样本)")
    estimate_bio_evaluation_cost(10)
    
    print("\n阶段2: 科学验证 (50样本)")
    estimate_bio_evaluation_cost(50)
    
    print("\n阶段3: 完整评估 (100样本)")
    estimate_bio_evaluation_cost(100)
    
    # 策略2: 采样策略
    print("\n📋 策略2: 智能采样")
    print("- 随机采样: 保证代表性")
    print("- 分层采样: 按长度/复杂度分层")
    print("- 困难样本优先: 筛选有挑战性的样本")
    
    # 策略3: 成本控制
    print("\n📋 策略3: 成本控制建议")
    print("✅ 推荐方案:")
    print("  - 科学验证: 30-50个精选样本")
    print("  - 成本控制: 预算$5-10 USD")
    print("  - 采样方法: 分层随机采样")
    print("  - 备用方案: 使用便宜的模型(如qwen)替代GPT")
    
    return True

def sample_bio_data_strategically(input_file, output_file, sample_size=50, strategy='random', seed=42):
    """
    智能采样BIO数据
    
    Args:
        input_file: 输入BIO数据文件
        output_file: 输出采样后的文件
        sample_size: 采样数量
        strategy: 采样策略 ('random', 'stratified', 'difficult')
        seed: 随机种子
    """
    
    random.seed(seed)
    
    print(f"🎯 执行BIO数据采样")
    print(f"策略: {strategy}")
    print(f"样本数: {sample_size}")
    
    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    
    print(f"原始数据: {len(data)}条")
    
    if strategy == 'random':
        # 随机采样
        sampled_data = random.sample(data, min(sample_size, len(data)))
        
    elif strategy == 'stratified':
        # 按长度分层采样
        # 按生成答案长度分为短/中/长三层
        short_data = [item for item in data if len(item.get('generated_answer', '')) < 200]
        medium_data = [item for item in data if 200 <= len(item.get('generated_answer', '')) < 500]
        long_data = [item for item in data if len(item.get('generated_answer', '')) >= 500]
        
        # 按比例采样
        short_samples = min(sample_size // 3, len(short_data))
        medium_samples = min(sample_size // 3, len(medium_data))
        long_samples = min(sample_size - short_samples - medium_samples, len(long_data))
        
        sampled_data = (
            random.sample(short_data, short_samples) +
            random.sample(medium_data, medium_samples) +
            random.sample(long_data, long_samples)
        )
        
        print(f"分层采样: 短({short_samples}) + 中({medium_samples}) + 长({long_samples})")
        
    elif strategy == 'difficult':
        # 选择困难样本 (较长的生成答案，可能包含更多事实)
        data_with_length = [(item, len(item.get('generated_answer', ''))) for item in data]
        data_with_length.sort(key=lambda x: x[1], reverse=True)  # 按长度降序
        sampled_data = [item for item, _ in data_with_length[:sample_size]]
        
        print(f"困难采样: 选择最长的{sample_size}个样本")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 保存采样结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 采样完成: {len(sampled_data)}条数据已保存到 {output_file}")
    
    return sampled_data

def main():
    parser = argparse.ArgumentParser(description="BIO评估成本优化工具")
    parser.add_argument('--action', type=str, choices=['estimate', 'strategy', 'sample'], 
                       default='strategy', help='执行的操作')
    parser.add_argument('--input_file', type=str, help='输入BIO数据文件')
    parser.add_argument('--output_file', type=str, help='输出采样文件')
    parser.add_argument('--sample_size', type=int, default=50, help='采样数量')
    parser.add_argument('--strategy', type=str, choices=['random', 'stratified', 'difficult'],
                       default='stratified', help='采样策略')
    parser.add_argument('--num_samples', type=int, default=100, help='评估样本数（用于成本估算）')
    
    args = parser.parse_args()
    
    if args.action == 'estimate':
        estimate_bio_evaluation_cost(args.num_samples)
    elif args.action == 'strategy':
        create_bio_evaluation_strategy()
    elif args.action == 'sample':
        if not args.input_file or not args.output_file:
            print("错误: 采样操作需要指定 --input_file 和 --output_file")
            return
        sample_bio_data_strategically(
            args.input_file, args.output_file, 
            args.sample_size, args.strategy
        )

if __name__ == "__main__":
    main()

# 使用示例:
"""
# 1. 查看评估策略建议
python bio_cost_optimizer.py --action strategy

# 2. 估算成本
python bio_cost_optimizer.py --action estimate --num_samples 100

# 3. 执行智能采样
python bio_cost_optimizer.py --action sample \
    --input_file /workspace/conRAG/data/bio/splits/bio_eval_scientific.jsonl \
    --output_file /workspace/conRAG/data/bio/bio_eval_sampled_50.jsonl \
    --sample_size 50 \
    --strategy stratified
"""