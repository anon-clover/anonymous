#!/usr/bin/env python3
"""
科学的数据划分脚本
支持多种实验设计方案
"""

import json
import argparse
import os
import random
from pathlib import Path

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    """保存JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def split_dataset(data, train_ratio=0.8, seed=42):
    """将数据集划分为训练集和测试集"""
    random.seed(seed)
    
    # 打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 计算划分点
    split_point = int(len(shuffled_data) * train_ratio)
    
    train_data = shuffled_data[:split_point]
    eval_data = shuffled_data[split_point:]
    
    return train_data, eval_data

def create_consensus_training_data(train_data_dict, output_file, mix_ratio=None):
    """
    创建混合的共识器训练数据
    
    Args:
        train_data_dict: {dataset_name: train_data}
        output_file: 输出文件路径
        mix_ratio: 混合比例，如 {"popqa": 0.6, "arc_challenge": 0.2, "bio": 0.2}
    """
    if mix_ratio is None:
        # 默认等权重混合
        mix_ratio = {name: 1.0/len(train_data_dict) for name in train_data_dict.keys()}
    
    # 计算每个数据集的样本数
    total_samples = sum(len(data) for data in train_data_dict.values())
    mixed_data = []
    
    for dataset_name, data in train_data_dict.items():
        ratio = mix_ratio.get(dataset_name, 0)
        if ratio > 0:
            # 随机采样
            sample_size = min(len(data), int(total_samples * ratio))
            sampled = random.sample(data, sample_size)
            
            # 添加数据集标识
            for item in sampled:
                item['dataset_source'] = dataset_name
            
            mixed_data.extend(sampled)
    
    # 打乱混合后的数据
    random.shuffle(mixed_data)
    
    # 保存为共识器训练格式
    save_jsonl(mixed_data, output_file)
    
    print(f"混合训练数据已保存: {output_file}")
    print(f"总样本数: {len(mixed_data)}")
    for dataset_name in train_data_dict.keys():
        count = sum(1 for item in mixed_data if item.get('dataset_source') == dataset_name)
        print(f"  {dataset_name}: {count} 样本")

def main():
    parser = argparse.ArgumentParser(description="科学的数据划分工具")
    parser.add_argument('--base_dir', type=str, default='/workspace/conRAG', 
                       help='项目根目录')
    parser.add_argument('--train_ratio', type=float, default=0.5,
                       help='训练集比例 (默认: 0.5)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子')
    parser.add_argument('--datasets', nargs='+',
                       default=['popqa', 'arc_challenge', 'bio', 'pubqa'],
                       help='要处理的数据集')
    parser.add_argument('--experiment_type', type=str, 
                       choices=['scientific', 'mixed', 'both'],
                       default='both',
                       help='实验类型: scientific(严格5:5), mixed(混合训练), both(两种都做)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    print(f"=== 科学数据划分工具 ===")
    print(f"训练比例: {args.train_ratio}")
    print(f"随机种子: {args.seed}")
    print(f"实验类型: {args.experiment_type}")
    print(f"处理数据集: {args.datasets}")
    
    # 存储所有数据集的划分结果
    all_train_data = {}
    all_eval_data = {}
    all_full_data = {}
    
    # 处理每个数据集
    for dataset in args.datasets:
        print(f"\n--- 处理数据集: {dataset} ---")
        
        # 输入文件路径
        input_file = f"{args.base_dir}/data/{dataset}/{dataset}_retrieved.jsonl"
        
        if not os.path.exists(input_file):
            print(f"警告: 文件不存在 {input_file}")
            continue
        
        # 加载数据
        data = load_jsonl(input_file)
        print(f"加载数据: {len(data)} 条")
        
        # 划分数据
        train_data, eval_data = split_dataset(data, args.train_ratio, args.seed)
        
        print(f"训练集: {len(train_data)} 条")
        print(f"测试集: {len(eval_data)} 条")
        
        # 保存路径
        output_dir = f"{args.base_dir}/data/{dataset}/splits"
        
        if args.experiment_type in ['scientific', 'both']:
            # 严格科学划分
            train_file = f"{output_dir}/{dataset}_train_scientific.jsonl"
            eval_file = f"{output_dir}/{dataset}_eval_scientific.jsonl"
            
            save_jsonl(train_data, train_file)
            save_jsonl(eval_data, eval_file)
            
            print(f"科学划分已保存:")
            print(f"  训练集: {train_file}")
            print(f"  测试集: {eval_file}")
        
        if args.experiment_type in ['mixed', 'both']:
            # 保存完整数据用于"应用场景"评估
            full_file = f"{output_dir}/{dataset}_full_for_application.jsonl"
            save_jsonl(data, full_file)
            print(f"完整数据: {full_file}")
        
        # 存储用于混合训练
        all_train_data[dataset] = train_data
        all_eval_data[dataset] = eval_data
        all_full_data[dataset] = data
    
    # 创建混合训练数据
    if args.experiment_type in ['mixed', 'both'] and all_train_data:
        print(f"\n--- 创建混合共识器训练数据 ---")
        
        # 推荐的混合比例
        mix_ratio = {
            'popqa': 0.4,
            'arc_challenge': 0.2,
            'bio': 0.2,
            'pubqa': 0.2
        }
        
        # 只使用实际存在的数据集
        actual_mix_ratio = {k: v for k, v in mix_ratio.items() if k in all_train_data}
        
        # 归一化比例
        total_ratio = sum(actual_mix_ratio.values())
        actual_mix_ratio = {k: v/total_ratio for k, v in actual_mix_ratio.items()}
        
        mixed_output = f"{args.base_dir}/data/consensus_training/mixed_consensus_training_data.jsonl"
        create_consensus_training_data(all_train_data, mixed_output, actual_mix_ratio)
    
    print(f"\n=== 数据划分完成 ===")
    print(f"建议的实验流程:")
    print(f"1. 使用混合训练数据训练共识器")
    print(f"2. 在科学测试集上验证方法有效性")
    print(f"3. 在完整数据集上展示应用效果")

if __name__ == "__main__":
    main()