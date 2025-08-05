#!/usr/bin/env python3


import json
import argparse
import os
from pathlib import Path

def load_jsonl(file_path):

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def extract_eval_retrieved_data(eval_file, retrieved_file, output_file):

    eval_data = load_jsonl(eval_file)
    print(f"{len(eval_data)} ")
    

    retrieved_data = load_jsonl(retrieved_file)
    print(f" {len(retrieved_data)} ")

    eval_queries = set()
    for item in eval_data:
        query = item.get('query') or item.get('question', '')
        if query:
            eval_queries.add(query.strip())
    

    eval_retrieved = []
    for item in retrieved_data:
        query = item.get('query') or item.get('question', '')
        if query and query.strip() in eval_queries:
            eval_retrieved.append(item)
    
    print(f" {len(eval_retrieved)} ")
    

    save_jsonl(eval_retrieved, output_file)
    print(f"{output_file}")

def main():
    parser = argparse.ArgumentParser(description="retrieved data")
    parser.add_argument('--base_dir', type=str, default='/workspace/conRAG',
                       help='context')
    parser.add_argument('--datasets', nargs='+',
                       default=['popqa', 'arc_challenge', 'bio'],
                       help='processing dataset')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    for dataset in args.datasets:
        print(f"\n {dataset}")
        

        eval_file = base_dir / 'data' / dataset / 'splits' / f'{dataset}_eval_scientific.jsonl'
        retrieved_file = base_dir / 'data' / dataset / f'{dataset}_retrieved.jsonl'
        output_file = base_dir / 'data' / dataset / 'splits' / f'{dataset}_eval_retrieved.jsonl'
        
        if not eval_file.exists():
            print(f"warning: {eval_file}")
            continue
            
        if not retrieved_file.exists():
            print(f"warning: {retrieved_file}")
            continue
        
        extract_eval_retrieved_data(eval_file, retrieved_file, output_file)

if __name__ == '__main__':
    main()
