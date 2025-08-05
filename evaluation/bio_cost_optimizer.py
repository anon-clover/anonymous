
import argparse
import json
import os
from tqdm import tqdm
import random

def estimate_bio_evaluation_cost(num_samples, avg_facts_per_sample=5):

    calls_per_sample = 1 + avg_facts_per_sample 
    total_calls = num_samples * calls_per_sample

    extract_tokens_per_call = 1000  # generate_atomic_facts max_tokens
    verify_tokens_per_call = 10     # verify_atomic_fact max_tokens
    
    total_extract_tokens = num_samples * extract_tokens_per_call
    total_verify_tokens = num_samples * avg_facts_per_sample * verify_tokens_per_call
    total_tokens = total_extract_tokens + total_verify_tokens
    
 
    estimated_cost_usd = total_tokens * 0.002 / 1000
    
    
    return total_calls, total_tokens, estimated_cost_usd

def create_bio_evaluation_strategy(): 
    estimate_bio_evaluation_cost(10)
    
    estimate_bio_evaluation_cost(50)

    estimate_bio_evaluation_cost(100)

    
    return True

def sample_bio_data_strategically(input_file, output_file, sample_size=50, strategy='random', seed=42):

    
    random.seed(seed)


    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    

    
    if strategy == 'random':

        sampled_data = random.sample(data, min(sample_size, len(data)))
        
    elif strategy == 'stratified':

        short_data = [item for item in data if len(item.get('generated_answer', '')) < 200]
        medium_data = [item for item in data if 200 <= len(item.get('generated_answer', '')) < 500]
        long_data = [item for item in data if len(item.get('generated_answer', '')) >= 500]
        

        short_samples = min(sample_size // 3, len(short_data))
        medium_samples = min(sample_size // 3, len(medium_data))
        long_samples = min(sample_size - short_samples - medium_samples, len(long_data))
        
        sampled_data = (
            random.sample(short_data, short_samples) +
            random.sample(medium_data, medium_samples) +
            random.sample(long_data, long_samples)
        )
        
    elif strategy == 'difficult':
        
        data_with_length = [(item, len(item.get('generated_answer', ''))) for item in data]
        data_with_length.sort(key=lambda x: x[1], reverse=True)  # 按长度降序
        sampled_data = [item for item, _ in data_with_length[:sample_size]]
   
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    
    return sampled_data

def main():
    parser = argparse.ArgumentParser(description="BIO")
    parser.add_argument('--action', type=str, choices=['estimate', 'strategy', 'sample'], 
                       default='strategy', help='')
    parser.add_argument('--input_file', type=str, help='')
    parser.add_argument('--output_file', type=str, help='')
    parser.add_argument('--sample_size', type=int, default=50, help='')
    parser.add_argument('--strategy', type=str, choices=['random', 'stratified', 'difficult'],
                       default='stratified', help='')
    parser.add_argument('--num_samples', type=int, default=100, help='')
    
    args = parser.parse_args()
    
    if args.action == 'estimate':
        estimate_bio_evaluation_cost(args.num_samples)
    elif args.action == 'strategy':
        create_bio_evaluation_strategy()
    elif args.action == 'sample':
        if not args.input_file or not args.output_file:
      
            return
        sample_bio_data_strategically(
            args.input_file, args.output_file, 
            args.sample_size, args.strategy
        )

if __name__ == "__main__":
    main()

python bio_cost_optimizer.py --action strategy

python bio_cost_optimizer.py --action estimate --num_samples 100

python bio_cost_optimizer.py --action sample \
    --input_file /workspace/conRAG/data/bio/splits/bio_eval_scientific.jsonl \
    --output_file /workspace/conRAG/data/bio/bio_eval_sampled_50.jsonl \
    --sample_size 50 \
    --strategy stratified
