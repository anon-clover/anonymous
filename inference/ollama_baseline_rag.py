#!/usr/bin/env python3
"""
Ollama完整共识RAG - 统一推理策略版本
使用Llama模型进行完整共识推理，结合内部知识和检索信息
采用统一的推理策略框架，提供更一致和高效的推理效果
"""

import json
import argparse
import os
import sys
from tqdm import tqdm
import requests
import time
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.utils import postprocess_popqa_answer, postprocess_pubqa_answer
from core.arc_utils import get_arc_choices, format_arc_choices_for_instruction, postprocess_arc_answer_unified, setup_arc_processing

# 导入统一推理策略
from inference.unified_reasoning_strategies import OllamaConsensusStrategy

TASK_INST = {
    "popqa": "Answer the following question based on the provided information.",
    "arc_challenge": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "bio": "Generate a comprehensive biography based on the provided information.",
    "pubqa": "Is the following statement correct or not? Say true if it's correct; otherwise say false."
}

# 使用统一推理策略的prompt函数
def format_enhanced_adaptive_prompt_llama(item_index, task, query, consensus_text, additional_evidence, choices_data=None, item_choices=None):
    """使用统一推理策略生成prompt"""
    return OllamaConsensusStrategy.format_enhanced_adaptive_prompt_llama(
        item_index, task, query, consensus_text, additional_evidence, choices_data, item_choices
    )


# 删除自我修正函数，改为单步推理对齐baseline方法

def call_ollama_api(prompt, base_url, model_name, max_tokens=150, temperature=0.0, max_retries=3, timeout=60):
    """调用Ollama API生成回答"""
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 1.0,
            "num_predict": max_tokens,
            "num_ctx": 4096,
            "seed": 42
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if 'response' in result:
                return result['response'].strip()
            else:
                print(f"Warning: Unexpected response format: {result}")
                return "Error: Invalid response format"
                
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                continue
            return "Error: Request timeout"
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                continue
            return f"Error: {str(e)}"
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                continue
            return f"Error: {str(e)}"
    
    return "Error: All retry attempts failed"

def postprocess_answer(answer, task):
    """修复后的答案后处理 - 更准确的答案提取"""
    answer = answer.strip()
    
    if task == "arc_challenge":
        return postprocess_arc_answer_unified(answer)
        
    elif task == "popqa":
        return postprocess_popqa_answer(answer)
    elif task == "pubqa":
        return postprocess_pubqa_answer(answer)
    
    return answer

def main():
    parser = argparse.ArgumentParser(description="Full Dataset Consensus RAG with Ollama")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--ollama_base_url', type=str, required=True)
    parser.add_argument('--ollama_model_name', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, choices=['popqa', 'arc_challenge', 'bio', 'pubqa'])
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--max_tokens', type=int, default=150, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for generation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f"=== Enhanced Adaptive RAG with Ollama ===")
    print(f"Task: {args.task}")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Model: {args.ollama_model_name}")
    print(f"API: {args.ollama_base_url}")
    print(f"Strategy: Enhanced adaptive reasoning based on information quality")
    
    # 使用统一的ARC处理设置
    choices_data = setup_arc_processing(args.input_file, args.task)
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    if args.num_samples > 0:
        input_data = input_data[:args.num_samples]
        print(f"Processing first {args.num_samples} samples")
    
    print(f"Total samples to process: {len(input_data)}")
    
    results = []
    for i, item in enumerate(tqdm(input_data, desc="Processing")):
        try:
            query = item.get('query', item.get('question', ''))
            if not query:
                print(f"Warning: No query found in item {i}")
                continue
            
            # 获取共识和额外证据
            consensus_text = item.get('consensus', '')
            additional_evidence = item.get('additional_evidence', [])
            item_choices = item.get('choices', None)

            # 增强自适应推理：根据信息质量调整策略
            prompt = format_enhanced_adaptive_prompt_llama(
                i, args.task, query, consensus_text, additional_evidence,
                choices_data, item_choices
            )
            generated_answer = call_ollama_api(
                prompt, args.ollama_base_url, args.ollama_model_name,
                args.max_tokens, args.temperature
            )

            # 后处理答案
            final_answer = postprocess_answer(generated_answer, args.task)

            results.append({
                'item_index': i,
                'query': query,
                'generated_answer': generated_answer,
                'processed_answer': final_answer,
                'method': 'ollama_consensus_unified',
                'model': args.ollama_model_name
            })
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            traceback.print_exc()
            continue
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Consensus-Only RAG with Ollama completed!")
    print(f"Processed {len(results)} samples")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
