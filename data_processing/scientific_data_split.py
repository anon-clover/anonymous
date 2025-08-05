
import json
import argparse
import os
import random
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

def split_dataset(data, train_ratio=0.8, seed=42):

    random.seed(seed)

    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    

    split_point = int(len(shuffled_data) * train_ratio)
    
    train_data = shuffled_data[:split_point]
    eval_data = shuffled_data[split_point:]
    
    return train_data, eval_data

def create_consensus_training_data(train_data_dict, output_file, mix_ratio=None):

    if mix_ratio is None:

        mix_ratio = {name: 1.0/len(train_data_dict) for name in train_data_dict.keys()}
    

    total_samples = sum(len(data) for data in train_data_dict.values())
    mixed_data = []
    
    for dataset_name, data in train_data_dict.items():
        ratio = mix_ratio.get(dataset_name, 0)
        if ratio > 0:
         
            sample_size = min(len(data), int(total_samples * ratio))
            sampled = random.sample(data, sample_size)
            
     
            for item in sampled:
                item['dataset_source'] = dataset_name
            
            mixed_data.extend(sampled)

    random.shuffle(mixed_data)
    

    save_jsonl(mixed_data, output_file)
    
    print(f"{output_file}")
    print(f"{len(mixed_data)}")
    for dataset_name in train_data_dict.keys():
        count = sum(1 for item in mixed_data if item.get('dataset_source') == dataset_name)
        print(f"  {dataset_name}: {count} 样本")

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--base_dir', type=str, default='/workspace/conRAG', 
                       help='')
    parser.add_argument('--train_ratio', type=float, default=0.5,
                       help='')
    parser.add_argument('--seed', type=int, default=42, 
                       help='')
    parser.add_argument('--datasets', nargs='+',
                       default=['popqa', 'arc_challenge', 'bio', 'pubqa'],
                       help='')
    parser.add_argument('--experiment_type', type=str, 
                       choices=['scientific', 'mixed', 'both'],
                       default='both',
                       help='')
    
    args = parser.parse_args()

    random.seed(args.seed)

    all_train_data = {}
    all_eval_data = {}
    all_full_data = {}
    

    for dataset in args.datasets:

        input_file = f"{args.base_dir}/data/{dataset}/{dataset}_retrieved.jsonl"
        
        if not os.path.exists(input_file):
            print(f"warning {input_file}")
            continue
    
        data = load_jsonl(input_file)
     
        train_data, eval_data = split_dataset(data, args.train_ratio, args.seed)

        output_dir = f"{args.base_dir}/data/{dataset}/splits"
        
        if args.experiment_type in ['scientific', 'both']:
            train_file = f"{output_dir}/{dataset}_train_scientific.jsonl"
            eval_file = f"{output_dir}/{dataset}_eval_scientific.jsonl"
            
            save_jsonl(train_data, train_file)
            save_jsonl(eval_data, eval_file)
        
        if args.experiment_type in ['mixed', 'both']:
            full_file = f"{output_dir}/{dataset}_full_for_application.jsonl"
            save_jsonl(data, full_file)

        all_train_data[dataset] = train_data
        all_eval_data[dataset] = eval_data
        all_full_data[dataset] = data
    

    if args.experiment_type in ['mixed', 'both'] and all_train_data:

        mix_ratio = {
            'popqa': 0.4,
            'arc_challenge': 0.2,
            'bio': 0.2,
            'pubqa': 0.2
        }
        

        actual_mix_ratio = {k: v for k, v in mix_ratio.items() if k in all_train_data}

        total_ratio = sum(actual_mix_ratio.values())
        actual_mix_ratio = {k: v/total_ratio for k, v in actual_mix_ratio.items()}
        
        mixed_output = f"{args.base_dir}/data/consensus_training/mixed_consensus_training_data.jsonl"
        create_consensus_training_data(all_train_data, mixed_output, actual_mix_ratio)
    

if __name__ == "__main__":
    main()
