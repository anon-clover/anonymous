#!/usr/bin/env python3
"""
Ollama智能RAG推理 - 统一推理策略版本
使用Llama模型进行自适应推理，结合内部知识和检索信息
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
from inference.unified_reasoning_strategies import OllamaIntelligentStrategy

TASK_INST = {
    "popqa": "Answer the following question based on your knowledge and any provided information.",
    "arc_challenge": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "bio": "Generate a comprehensive biography based on your knowledge and any provided information.",
    "pubqa": "Is the following statement correct or not? Say true if it's correct; otherwise say false."
}

# 使用统一推理策略的prompt函数
def format_selfrag_inspired_adaptive_prompt(item_index, task, query, consensus_text, additional_evidence, choices_data=None, item_choices=None):
    """使用统一推理策略生成prompt"""
    return OllamaIntelligentStrategy.format_selfrag_inspired_adaptive_prompt(
        item_index, task, query, consensus_text, additional_evidence, choices_data, item_choices
    )


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
    """
    答案后处理 - 提取并清理答案
    """
    # 基本清理
    answer = answer.replace("</s>", "").replace("<|endoftext|>", "").strip()

    # 如果答案包含推理过程，尝试提取最终答案
    # 查找常见的答案标记
    answer_markers = [
        "my answer is:",
        "the answer is:",
        "therefore:",
        "in conclusion:",
        "final answer:",
        "answer:"
    ]

    for marker in answer_markers:
        if marker in answer.lower():
            # 找到标记后的内容
            idx = answer.lower().rfind(marker)
            if idx != -1:
                answer = answer[idx + len(marker):].strip()
                # 只取第一行作为答案（如果后面还有解释）
                if '\n' in answer:
                    answer = answer.split('\n')[0].strip()
                break

    if task == "arc_challenge":
        return postprocess_arc_answer_unified(answer)
    elif task == "popqa":
        return postprocess_popqa_answer(answer)
    elif task == "pubqa":
        return postprocess_pubqa_answer(answer)

    return answer

def main():
    parser = argparse.ArgumentParser(description="Adaptive RAG with Ollama - Natural language prompting for combining internal knowledge and retrieval")
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
    
    print(f"=== Adaptive RAG with Ollama (Natural Language) ===")
    print(f"Task: {args.task}")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Model: {args.ollama_model_name}")
    print(f"API: {args.ollama_base_url}")
    
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

            # 使用Self-RAG启发的自适应推理prompt
            prompt = format_selfrag_inspired_adaptive_prompt(
                i, args.task, query, consensus_text, additional_evidence,
                choices_data, item_choices
            )
            generated_answer = call_ollama_api(
                prompt, args.ollama_base_url, args.ollama_model_name,
                args.max_tokens, args.temperature
            )

            # 后处理答案
            final_answer = postprocess_answer(generated_answer, args.task)

            # 保存结果 - 使用与第一个脚本一致的字段名
            result = {
                'query': query,
                'processed_answer': final_answer,  # 评估脚本期望的字段名
                'raw_selfrag_response': generated_answer,  # 保持字段名一致
                'method': 'ollama_intelligent_unified',
                'model': args.ollama_model_name
            }

            # 保留原始字段
            for key in ['id', 'answerKey', 'choices', 'choices_data']:
                if key in item:
                    result[key] = item[key]

            results.append(result)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            traceback.print_exc()
            # 添加错误结果
            results.append({
                'query': query,
                'processed_answer': '',
                'raw_selfrag_response': f'Error: {str(e)}',
                'method': 'ollama_intelligent_unified',
                'model': args.ollama_model_name
            })
            continue
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Adaptive RAG with Ollama (Natural Language) completed!")
    print(f"Processed {len(results)} samples")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
