"""
数据质量分析和清理工具

用于分析训练数据质量并生成清理后的高质量训练数据
"""

import json
import argparse
import os
from typing import List, Dict, Any, Tuple
import re
from collections import Counter

def analyze_consensus_quality(file_path: str) -> Dict[str, Any]:
    """分析共识数据的质量"""
    total_count = 0
    low_quality_count = 0
    empty_consensus = 0
    short_consensus = 0
    valid_consensus = 0
    
    low_quality_patterns = [
        "do not contain sufficient evidence",
        "cannot be generated", 
        "no consensus answer",
        "insufficient evidence",
        "提供的文档不包含",
        "无法总结",
        "没有足够证据",
        "NO_PASSAGES_PROVIDED",
        "PASSAGES_EMPTY",
        "ERROR_PROCESSING",
    ]
    
    quality_issues = Counter()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # 支持两种格式：JSONL和文本格式
                if line.strip().startswith('{'):
                    # JSONL格式
                    data = json.loads(line.strip())
                    consensus = data.get('consensus', '')
                else:
                    # 文本格式
                    if line.startswith('consensus:'):
                        consensus = line.split('consensus:', 1)[1].strip()
                    else:
                        continue
                        
                total_count += 1
                
                if not consensus:
                    empty_consensus += 1
                    quality_issues['empty'] += 1
                    continue
                
                if len(consensus.strip()) < 20:
                    short_consensus += 1
                    quality_issues['too_short'] += 1
                
                # 检查低质量模式
                consensus_lower = consensus.lower()
                is_low_quality = False
                for pattern in low_quality_patterns:
                    if pattern.lower() in consensus_lower:
                        low_quality_count += 1
                        quality_issues[f'pattern_{pattern[:20]}'] += 1
                        is_low_quality = True
                        break
                
                if not is_low_quality and len(consensus.strip()) >= 20:
                    valid_consensus += 1
                    
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    return {
        'total_count': total_count,
        'valid_consensus': valid_consensus,
        'low_quality_count': low_quality_count,
        'empty_consensus': empty_consensus,
        'short_consensus': short_consensus,
        'quality_issues': dict(quality_issues),
        'quality_rate': valid_consensus / total_count if total_count > 0 else 0
    }

def clean_training_data(input_file: str, output_file: str, min_consensus_length: int = 20) -> int:
    """清理训练数据，移除低质量样本"""
    
    low_quality_patterns = [
        "do not contain sufficient evidence",
        "cannot be generated",
        "no consensus answer", 
        "insufficient evidence",
        "提供的文档不包含",
        "无法总结",
        "没有足够证据",
        "NO_PASSAGES_PROVIDED",
        "PASSAGES_EMPTY",
        "ERROR_PROCESSING",
    ]
    
    cleaned_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        current_entry = {}
        consensus_lines = []
        
        for line in infile:
            line = line.strip()
            total_count += 1
            
            if line.startswith('query:'):
                # 处理前一个条目
                if current_entry and consensus_lines:
                    consensus = ' '.join(consensus_lines).strip()
                    if is_high_quality_consensus(consensus, low_quality_patterns, min_consensus_length):
                        # 写入高质量条目
                        outfile.write(f"query: {current_entry['query']}\n")
                        outfile.write(f"documents: {current_entry['documents']}\n")
                        outfile.write(f"consensus: {consensus}\n\n")
                        cleaned_count += 1
                
                # 开始新条目
                current_entry = {'query': line.split('query:', 1)[1].strip()}
                consensus_lines = []
                
            elif line.startswith('documents:'):
                current_entry['documents'] = line.split('documents:', 1)[1].strip()
                
            elif line.startswith('consensus:'):
                consensus_lines = [line.split('consensus:', 1)[1].strip()]
                
            elif line and consensus_lines:  # 续行
                consensus_lines.append(line)
        
        # 处理最后一个条目
        if current_entry and consensus_lines:
            consensus = ' '.join(consensus_lines).strip()
            if is_high_quality_consensus(consensus, low_quality_patterns, min_consensus_length):
                outfile.write(f"query: {current_entry['query']}\n")
                outfile.write(f"documents: {current_entry['documents']}\n")
                outfile.write(f"consensus: {consensus}\n\n")
                cleaned_count += 1
    
    print(f"Cleaned {cleaned_count} high-quality samples from {total_count} total samples")
    print(f"Quality rate: {cleaned_count/total_count*100:.1f}%")
    
    return cleaned_count

def is_high_quality_consensus(consensus: str, low_quality_patterns: List[str], min_length: int) -> bool:
    """判断共识是否高质量"""
    if not consensus or len(consensus.strip()) < min_length:
        return False
    
    consensus_lower = consensus.lower()
    for pattern in low_quality_patterns:
        if pattern.lower() in consensus_lower:
            return False
    
    return True

def generate_enhanced_training_data(input_files: List[str], output_file: str, 
                                   dataset_weights: Dict[str, float] = None) -> None:
    """生成增强的混合训练数据"""
    
    if dataset_weights is None:
        dataset_weights = {
            'popqa': 0.4,
            'arc_challenge': 0.2, 
            'bio': 0.2,
            'pubqa': 0.2
        }
    
    all_high_quality_data = []
    
    for input_file in input_files:
        dataset_name = None
        for name in dataset_weights.keys():
            if name in input_file.lower():
                dataset_name = name
                break
        
        if not dataset_name:
            dataset_name = 'unknown'
        
        print(f"Processing {dataset_name} data from {input_file}")
        
        # 临时清理文件
        temp_file = input_file + '.cleaned'
        cleaned_count = clean_training_data(input_file, temp_file)
        
        # 读取清理后的数据
        dataset_data = []
        with open(temp_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            entries = content.split('\n\n')
            
            for entry in entries:
                if entry.strip():
                    lines = entry.strip().split('\n')
                    if len(lines) >= 3:
                        query_line = next((l for l in lines if l.startswith('query:')), '')
                        doc_line = next((l for l in lines if l.startswith('documents:')), '')
                        consensus_line = next((l for l in lines if l.startswith('consensus:')), '')
                        
                        if query_line and doc_line and consensus_line:
                            dataset_data.append({
                                'query': query_line.split('query:', 1)[1].strip(),
                                'documents': doc_line.split('documents:', 1)[1].strip(),
                                'consensus': consensus_line.split('consensus:', 1)[1].strip(),
                                'dataset_source': dataset_name
                            })
        
        print(f"Loaded {len(dataset_data)} high-quality samples from {dataset_name}")
        all_high_quality_data.extend(dataset_data)
        
        # 清理临时文件
        os.remove(temp_file)
    
    # 根据权重采样
    import random
    random.seed(42)
    
    final_training_data = []
    total_samples = len(all_high_quality_data)
    
    for dataset_name, weight in dataset_weights.items():
        dataset_samples = [d for d in all_high_quality_data if d['dataset_source'] == dataset_name]
        if dataset_samples:
            target_count = int(total_samples * weight)
            if len(dataset_samples) > target_count:
                selected = random.sample(dataset_samples, target_count)
            else:
                selected = dataset_samples
            final_training_data.extend(selected)
            print(f"Selected {len(selected)} samples from {dataset_name}")
    
    # 打乱顺序
    random.shuffle(final_training_data)
    
    # 保存为JSONL格式（用于新的训练脚本）
    jsonl_output = output_file.replace('.txt', '.jsonl')
    with open(jsonl_output, 'w', encoding='utf-8') as f:
        for item in final_training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存为文本格式（兼容现有训练脚本）
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in final_training_data:
            f.write(f"query: {item['query']}\n")
            f.write(f"documents: {item['documents']}\n")
            f.write(f"consensus: {item['consensus']}\n\n")
    
    print(f"Enhanced training data saved to {output_file} and {jsonl_output}")
    print(f"Total high-quality samples: {len(final_training_data)}")

def main():
    parser = argparse.ArgumentParser(description="Training data quality analysis and cleaning")
    parser.add_argument('--action', type=str, required=True,
                       choices=['analyze', 'clean', 'enhance'],
                       help='Action to perform')
    parser.add_argument('--input_file', type=str,
                       help='Input file for analyze/clean actions')
    parser.add_argument('--input_files', nargs='+',
                       help='Input files for enhance action')
    parser.add_argument('--output_file', type=str,
                       help='Output file for clean/enhance actions')
    parser.add_argument('--min_consensus_length', type=int, default=20,
                       help='Minimum consensus length for quality filtering')
    
    args = parser.parse_args()
    
    if args.action == 'analyze':
        if not args.input_file:
            print("--input_file required for analyze action")
            return
        
        print(f"Analyzing consensus quality in {args.input_file}")
        quality_stats = analyze_consensus_quality(args.input_file)
        
        print("\n=== Quality Analysis Results ===")
        print(f"Total samples: {quality_stats['total_count']}")
        print(f"Valid consensus: {quality_stats['valid_consensus']}")
        print(f"Low quality: {quality_stats['low_quality_count']}")
        print(f"Empty consensus: {quality_stats['empty_consensus']}")
        print(f"Too short: {quality_stats['short_consensus']}")
        print(f"Quality rate: {quality_stats['quality_rate']:.1%}")
        
        print("\n=== Quality Issues Breakdown ===")
        for issue, count in quality_stats['quality_issues'].items():
            print(f"{issue}: {count}")
    
    elif args.action == 'clean':
        if not args.input_file or not args.output_file:
            print("--input_file and --output_file required for clean action")
            return
        
        clean_training_data(args.input_file, args.output_file, args.min_consensus_length)
    
    elif args.action == 'enhance':
        if not args.input_files or not args.output_file:
            print("--input_files and --output_file required for enhance action")
            return
        
        generate_enhanced_training_data(args.input_files, args.output_file)

if __name__ == "__main__":
    main()