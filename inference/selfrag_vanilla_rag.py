import json
import argparse
import os
from tqdm import tqdm
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('/workspace/conRAG')
from core.utils import format_arc_choices_for_prompt, postprocess_arc_answer, postprocess_popqa_answer, postprocess_pubqa_answer

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

def format_prompt(item_index, task, query, passages, choices_data=None, item_choices=None):
    instruction = TASK_INST[task] + "\n\n## Input:\n\n" + query

    if task == "arc_challenge":
        # 优先使用 item 中的 choices，然后是外部 choices_data
        choices_to_use = item_choices or choices_data
        if choices_to_use:
            if isinstance(choices_to_use, dict) and 'text' in choices_to_use and 'label' in choices_to_use:
                # 格式: {"text": [...], "label": ["A", "B", "C", "D"]}
                choices_formatted = ""
                for label, text in zip(choices_to_use['label'], choices_to_use['text']):
                    choices_formatted += f"\n{label}: {text}"
                instruction += choices_formatted
            else:
                # 旧格式兼容
                choices_text = format_arc_choices_for_prompt(choices_to_use)
                if choices_text:
                    choices_formatted = choices_text.replace("A)", "\nA: ").replace("B)", "\nB: ").replace("C)", "\nC: ").replace("D)", "\nD: ")
                    instruction += choices_formatted
    
    prompt = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)
    
    # Vanilla RAG: 使用原始检索的passages，但改进组织方式
    if passages and isinstance(passages, list) and len(passages) > 0:
        # 组织检索内容，让Self-RAG更好地利用
        prompt += "[Retrieval]"
        for i, passage in enumerate(passages[:5]):  # 取前5个最相关的检索文档
            if passage and passage.strip():
                prompt += "<paragraph>Passage {}: {}</paragraph>".format(i+1, passage.strip())
        
        # 让Self-RAG评估检索内容
        if task == "arc_challenge":
            prompt += "[Relevant]Based on the retrieved passages, I need to evaluate each answer choice A, B, C, and D. "
        elif task == "pubqa":
            prompt += "[Relevant]Based on the retrieved passages, I need to determine if the statement is true or false. "
        elif task == "popqa":
            prompt += "[Relevant]Based on the retrieved passages, the answer is: "
        elif task == "bio":
            prompt += "[Relevant]Based on the retrieved passages, I can generate the following biography: "
    else:
        prompt += "[No Retrieval]I will answer based on my knowledge. "
    
    return prompt

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
        processed = postprocess_arc_answer(answer)
        # 确保答案是A、B、C、D之一
        if processed not in ["A", "B", "C", "D"]:
            import re
            match = re.search(r'\b([A-D])\b', original_answer)
            if match:
                processed = match.group(1)
            else:
                first_char = answer[0].upper() if answer and answer[0].upper() in "ABCD" else "A"
                processed = first_char
        return processed
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

def query_selfrag(model, tokenizer, prompt, device, task, max_tokens=100, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 根据任务调整生成参数
    if task == "arc_challenge":
        actual_temp = 0.0
        actual_top_p = 1.0
        actual_max_tokens = min(max_tokens, 150)
    elif task == "pubqa":
        actual_temp = 0.1
        actual_top_p = 0.95
        actual_max_tokens = min(max_tokens, 200)
    elif task == "popqa":
        actual_temp = temperature
        actual_top_p = 0.9
        actual_max_tokens = max_tokens
    else:  # bio
        actual_temp = max(temperature, 0.2)
        actual_top_p = 0.9
        actual_max_tokens = max_tokens
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=actual_max_tokens,
            temperature=actual_temp,
            top_p=actual_top_p,
            do_sample=actual_temp > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0
        )
    
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--selfrag_model_path', type=str, default='/workspace/selfrag')
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
    
    choices_data = None
    if args.task == "arc_challenge":
        # 尝试多个可能的choices文件路径
        possible_choices_files = [
            args.input_file.replace('_enhanced_consensus_evidence.jsonl', '_choices.txt'),
            args.input_file.replace('enhanced_eval/', 'arc_challenge/splits/').replace('_enhanced_consensus_evidence.jsonl', '_choices.txt'),
            '/workspace/conRAG/data/arc_challenge/splits/arc_challenge_eval_scientific_choices.txt'
        ]
        
        for choices_file in possible_choices_files:
            if os.path.exists(choices_file):
                with open(choices_file, 'r') as f:
                    choices_data = [line.strip() for line in f.readlines()]
                print(f"Found choices file: {choices_file}")
                break
        
        if choices_data is None:
            print(f"Warning: No choices file found for arc_challenge task")
    
    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    if args.num_samples > 0:
        data = data[:args.num_samples]
    
    results = []
    for i, item in enumerate(tqdm(data)):
        query = item.get('query', '')
        passages = item.get('original_passages', item.get('passages', []))  # 使用原始检索的passages
        item_choices = item.get('choices', None)  # 从数据项中获取选项

        prompt = format_prompt(i, args.task, query, passages, choices_data, item_choices)
        raw_answer = query_selfrag(model, tokenizer, prompt, device, args.task, args.max_tokens, args.temperature)
        final_answer = postprocess_answer(raw_answer, args.task)
        
        results.append({
            "query": query,
            "raw_selfrag_response": raw_answer,
            "processed_answer": final_answer,
            "method": "selfrag_vanilla"
        })
    
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"SelfRAG Vanilla RAG完成处理 {len(results)} 个样本")

if __name__ == "__main__":
    main()
