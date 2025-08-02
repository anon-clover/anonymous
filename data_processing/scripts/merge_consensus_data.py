#!/usr/bin/env python3
"""
合并多个数据集的共识训练数据为混合训练数据
"""

import os
import random
import argparse
from pathlib import Path

def read_consensus_file(file_path):
    """读取共识训练数据文件"""
    data = []
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # 按照 "query:" 分割数据
        entries = content.split('query:')[1:]  # 跳过第一个空元素
        
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
                
            # 解析每个条目
            lines = entry.split('\n')
            query = ""
            documents = ""
            consensus = ""
            
            current_section = ""
            for line in lines:
                line = line.strip()
                if line.startswith('documents:'):
                    current_section = "documents"
                    documents = line[10:].strip()  # 去掉 "documents:" 前缀
                elif line.startswith('consensus:'):
                    current_section = "consensus"
                    consensus = line[10:].strip()  # 去掉 "consensus:" 前缀
                elif current_section == "documents":
                    documents += " " + line
                elif current_section == "consensus":
                    consensus += " " + line
                else:
                    # 第一行是query
                    query += " " + line
            
            # 清理数据
            query = query.strip()
            documents = documents.strip()
            consensus = consensus.strip()
            
            if query and documents and consensus and consensus != "CONSENSUS_GENERATION_FAILED":
                data.append({
                    'query': query,
                    'documents': documents,
                    'consensus': consensus
                })
    
    return data

def write_mixed_consensus_file(output_path, mixed_data):
    """写入混合共识训练数据"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in mixed_data:
            f.write(f"query: {item['query']}\n")
            f.write(f"documents: {item['documents']}\n")
            f.write(f"consensus: {item['consensus']}\n\n")

def main():
    parser = argparse.ArgumentParser(description="合并多个数据集的共识训练数据")
    parser.add_argument('--base_dir', type=str, default='/workspace/conRAG',
                       help='项目根目录')
    parser.add_argument('--datasets', nargs='+', 
                       default=['popqa', 'arc_challenge', 'bio', 'pubqa'],
                       help='要合并的数据集')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='每个数据集的样本数量')
    parser.add_argument('--output_file', type=str, 
                       default='/workspace/conRAG/data/consensus_training/mixed_consensus_training_data_gpt4o_2024_11_20.txt',
                       help='输出文件路径')
    parser.add_argument('--mix_ratio', type=str, default='0.4,0.2,0.2,0.2',
                       help='混合比例 (popqa,arc_challenge,bio,pubqa)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 解析混合比例
    mix_ratios = [float(x) for x in args.mix_ratio.split(',')]
    if len(mix_ratios) != len(args.datasets):
        print(f"错误: 混合比例数量({len(mix_ratios)})与数据集数量({len(args.datasets)})不匹配")
        return
    
    if abs(sum(mix_ratios) - 1.0) > 0.001:
        print(f"错误: 混合比例总和应为1.0，当前为{sum(mix_ratios)}")
        return
    
    print("=== 合并共识训练数据 ===")
    print(f"基础目录: {args.base_dir}")
    print(f"数据集: {args.datasets}")
    print(f"混合比例: {dict(zip(args.datasets, mix_ratios))}")
    print(f"输出文件: {args.output_file}")
    print()
    
    # 读取各数据集的共识数据
    all_data = {}
    total_samples = 0
    
    for dataset in args.datasets:
        input_file = f"{args.base_dir}/data/{dataset}/{dataset}_consensus_for_t5_training_gpt4o_2024_11_20_{args.num_samples}_samples.txt"
        print(f"读取 {dataset} 数据: {input_file}")
        
        data = read_consensus_file(input_file)
        all_data[dataset] = data
        
        print(f"  成功读取 {len(data)} 个样本")
        total_samples += len(data)
    
    print(f"\n总共读取 {total_samples} 个样本")
    
    # 按比例混合数据
    mixed_data = []
    
    for i, dataset in enumerate(args.datasets):
        data = all_data[dataset]
        ratio = mix_ratios[i]
        
        if not data:
            print(f"警告: {dataset} 没有可用数据，跳过")
            continue
        
        # 计算该数据集应该贡献的样本数
        target_samples = int(total_samples * ratio)
        
        # 如果数据不够，使用全部数据
        if len(data) <= target_samples:
            selected_data = data
        else:
            # 随机采样
            selected_data = random.sample(data, target_samples)
        
        # 添加数据集标识
        for item in selected_data:
            item['dataset_source'] = dataset
        
        mixed_data.extend(selected_data)
        print(f"{dataset}: 贡献 {len(selected_data)} 个样本 (目标: {target_samples})")
    
    # 随机打乱混合数据
    random.shuffle(mixed_data)
    
    print(f"\n混合后总样本数: {len(mixed_data)}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 写入混合数据
    write_mixed_consensus_file(args.output_file, mixed_data)
    
    print(f"✅ 混合共识训练数据已保存到: {args.output_file}")
    
    # 显示最终统计
    print("\n=== 最终数据分布 ===")
    dataset_counts = {}
    for item in mixed_data:
        source = item.get('dataset_source', 'unknown')
        dataset_counts[source] = dataset_counts.get(source, 0) + 1
    
    for dataset, count in dataset_counts.items():
        percentage = count / len(mixed_data) * 100
        print(f"{dataset}: {count} 样本 ({percentage:.1f}%)")
    
    print(f"\n下一步: 使用 {args.output_file} 训练T5共识模型")

if __name__ == "__main__":
    main()
