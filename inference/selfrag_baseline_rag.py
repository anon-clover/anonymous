import json
import argparse
import os
from tqdm import tqdm
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('/root/shh/FGRAG/fgrag')
from core.utils import format_arc_choices_for_prompt, postprocess_arc_answer, postprocess_popqa_answer, postprocess_pubqa_answer
from core.arc_utils import get_arc_choices, format_arc_choices_for_instruction, postprocess_arc_answer_unified, setup_arc_processing

TASK_INST = {
    "popqa": "Answer the following question based on the provided information.",
    "arc_challenge": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "bio": "Generate a comprehensive biography based on the provided information.",
    "pubqa": "Is the following statement correct or not? Say true if it's correct; otherwise say false."
}

control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", 
                  "[No Retrieval]", "[Retrieval]", "[Irrelevant]", "[Relevant]", 
                  "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", 
                  "[Utility:3]", "[Utility:4]", "[Utility:5]", "[Continue to Use Evidence]"]



def postprocess_answer(answer, task):
    # 保存原始答案用于调试
    original_answer = answer
    
    # 清理控制tokens
    for token in control_tokens:
        answer = answer.replace(token, "")
    answer = answer.replace("</s>", "").replace("<|endoftext|>", "").strip()
    answer = answer.replace("\n", " ").strip()
    
    # 任务特定的后处理
    if task == "arc_challenge":
        return postprocess_arc_answer_unified(answer, original_answer)
    elif task == "popqa":
        return postprocess_popqa_answer(answer)
    elif task == "pubqa":
        processed = postprocess_pubqa_answer(answer)
        if processed.lower() not in ["true", "false"]:
            if any(word in original_answer.lower() for word in ["true", "yes", "correct", "right"]):
                processed = "true"
            else:
                processed = "false"
        return processed
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--selfrag_model_path', type=str, default='/root/shh/FGRAG/fgrag')
    parser.add_argument('--task', type=str, required=True, choices=['popqa', 'arc_challenge', 'bio', 'pubqa'])
    parser.add_argument('--max_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--num_samples', type=int, default=-1)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(args.selfrag_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.selfrag_model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用统一的ARC处理设置
    choices_data = setup_arc_processing(args.input_file, args.task)
    
    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    if args.num_samples > 0:
        data = data[:args.num_samples]
    
    results = []
    for i, item in enumerate(tqdm(data)):
        query = item.get('query', '')
        consensus_text = item.get('consensus', '')  # 使用共识文本
        item_choices = item.get('choices', None)  # 从数据项中获取选项

        prompt = format_enhanced_adaptive_prompt(i, args.task, query, consensus_text, choices_data, item_choices)
        raw_answer = query_selfrag(model, tokenizer, prompt, device, args.task, args.max_tokens, args.temperature)
        final_answer = postprocess_answer(raw_answer, args.task)
        
        results.append({
            "query": query,
            "raw_selfrag_response": raw_answer,
            "processed_answer": final_answer,
            "method": "selfrag_baseline"
        })
    
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"SelfRAG Baseline (共识)完成处理 {len(results)} 个样本")

if __name__ == "__main__":
    main()
