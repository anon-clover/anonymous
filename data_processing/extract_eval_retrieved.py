#!/usr/bin/env python3
"""
从原始retrieved文件中提取评估集对应的数据
用于vanilla RAG的评估
"""

import json
import argparse
import os
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

def extract_eval_retrieved_data(eval_file, retrieved_file, output_file):
    """
    根据评估集提取对应的retrieved数据
    """
    # 加载评估集数据
    eval_data = load_jsonl(eval_file)
    print(f"加载评估集: {len(eval_data)} 条")
    
    # 加载原始retrieved数据
    retrieved_data = load_jsonl(retrieved_file)
    print(f"加载retrieved数据: {len(retrieved_data)} 条")
    
    # 提取评估集中的query/question
    eval_queries = set()
    for item in eval_data:
        query = item.get('query') or item.get('question', '')
        if query:
            eval_queries.add(query.strip())
    
    # 从retrieved中筛选出评估集对应的数据
    eval_retrieved = []
    for item in retrieved_data:
        query = item.get('query') or item.get('question', '')
        if query and query.strip() in eval_queries:
            eval_retrieved.append(item)
    
    print(f"提取到评估集对应的retrieved数据: {len(eval_retrieved)} 条")
    
    # 保存结果
    save_jsonl(eval_retrieved, output_file)
    print(f"已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="提取评估集对应的retrieved数据")
    parser.add_argument('--base_dir', type=str, default='/workspace/conRAG',
                       help='项目根目录')
    parser.add_argument('--datasets', nargs='+',
                       default=['popqa', 'arc_challenge', 'bio'],
                       help='要处理的数据集')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    for dataset in args.datasets:
        print(f"\n处理数据集: {dataset}")
        
        # 文件路径
        eval_file = base_dir / 'data' / dataset / 'splits' / f'{dataset}_eval_scientific.jsonl'
        retrieved_file = base_dir / 'data' / dataset / f'{dataset}_retrieved.jsonl'
        output_file = base_dir / 'data' / dataset / 'splits' / f'{dataset}_eval_retrieved.jsonl'
        
        if not eval_file.exists():
            print(f"警告: 评估集文件不存在: {eval_file}")
            continue
            
        if not retrieved_file.exists():
            print(f"警告: retrieved文件不存在: {retrieved_file}")
            continue
        
        extract_eval_retrieved_data(eval_file, retrieved_file, output_file)

if __name__ == '__main__':
    main()