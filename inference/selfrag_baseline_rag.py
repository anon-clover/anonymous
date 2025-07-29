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

def format_enhanced_adaptive_prompt(item_index, task, query, consensus_text, choices_data=None, item_choices=None):
    """
    增强自适应推理格式：根据信息质量调整推理策略
    - 检索信息充分支持时：更多依赖检索信息
    - 信息不全/不足时：更多相信内部知识
    """
    # 构建指令部分
    instruction = TASK_INST[task] + "\n\n## Input:\n\n" + query

    # 添加选项（对于ARC Challenge）
    if task == "arc_challenge":
        choices_to_use = get_arc_choices(item_choices, choices_data, item_index)
        instruction = format_arc_choices_for_instruction(choices_to_use, instruction)

    # 构建自适应推理的prompt
    prompt = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)

    # 判断是否有有效的证据
    has_valid_consensus = (consensus_text and consensus_text.strip() and
                          "ConsensusMissingInInput" not in consensus_text and
                          not any(marker in consensus_text.lower() for marker in ["no consensus answer", "insufficient evidenc"]))

    # 判断信息是否充分支持
    consensus_lower = consensus_text.lower() if consensus_text else ""
    is_information_insufficient = any(phrase in consensus_lower for phrase in
                                    ["do not contain", "do not directly address", "insufficient",
                                     "no information", "not mentioned", "does not provide"])

    # 判断信息是否明确支持/反驳
    has_clear_support = any(word in consensus_lower for word in
                           ['support', 'confirm', 'show', 'demonstrate', 'indicate', 'found', 'study'])
    has_clear_refute = any(word in consensus_lower for word in
                          ['contradict', 'refute', 'oppose', 'against', 'deny'])

    if has_valid_consensus:
        # 先让模型基于内部知识思考
        prompt += "[No Retrieval]Let me first consider what I know about this question from my training data.\n\n"

        # 然后提供检索信息
        prompt += "[Retrieval]I also have access to retrieved information:\n"
        prompt += "<paragraph>{0}</paragraph>".format(consensus_text.strip())

        # 根据信息质量调整推理策略
        if is_information_insufficient:
            # 信息不足时，更多相信内部知识
            prompt += "\n\nThe retrieved information appears to be insufficient. I should rely more on my training knowledge to answer this question. "
        elif has_clear_support or has_clear_refute:
            # 有明确支持/反驳时，更多依赖检索信息
            prompt += "\n\nThe retrieved information provides clear evidence. I should prioritize this evidence in my reasoning. "
        else:
            # 一般情况，平衡考虑
            prompt += "\n\nBased on both my knowledge and the retrieved information, "

    else:
        # 没有检索信息时，直接使用内部知识
        prompt += "[No Retrieval]"

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
