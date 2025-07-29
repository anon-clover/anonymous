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
    "popqa": "Answer the following question based on your knowledge and any provided information.",
    "arc_challenge": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "bio": "Generate a comprehensive biography based on your knowledge and any provided information.",
    "pubqa": "Is the following statement correct or not? Say true if it's correct; otherwise say false."
}

control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", 
                  "[No Retrieval]", "[Retrieval]", "[Irrelevant]", "[Relevant]", 
                  "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", 
                  "[Utility:3]", "[Utility:4]", "[Utility:5]", "[Continue to Use Evidence]"]

def format_adaptive_prompt(item_index, task, query, consensus_text, additional_evidence_list, choices_data=None, item_choices=None):
    """
    自适应推理格式：让模型自主判断是否使用检索信息
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

    has_valid_evidence = (additional_evidence_list and isinstance(additional_evidence_list, list) and
                         any(e and e.strip() and len(e.strip()) > 20 for e in additional_evidence_list))

    # 关键改进：让模型自主判断是否需要检索信息
    if has_valid_consensus or has_valid_evidence:
        # 先让模型基于内部知识思考
        prompt += "[No Retrieval]Let me first consider what I know about this question from my training data.\n\n"

        # 然后提供检索信息
        prompt += "[Retrieval]I also have access to retrieved information:\n"

        # 添加共识信息
        if has_valid_consensus:
            prompt += "<paragraph>{0}</paragraph>".format(consensus_text.strip())

        # 添加额外证据
        if has_valid_evidence:
            evidence_parts = []
            for evidence in additional_evidence_list[:2]:  # 最多2个证据
                if evidence and evidence.strip() and len(evidence.strip()) > 20:
                    evidence_parts.append(evidence.strip())

            if evidence_parts:
                prompt += "[Continue to Use Evidence]<paragraph>{0}</paragraph>".format(' '.join(evidence_parts))

        # 让模型综合判断
        prompt += "\n\nBased on both my knowledge and the retrieved information, "

    else:
        # 没有检索信息时，直接使用内部知识
        prompt += "[No Retrieval]"

    return prompt

def format_enhanced_adaptive_prompt(item_index, task, query, consensus_text, additional_evidence_list, choices_data=None, item_choices=None):
    """
    增强的自适应推理格式：根据信息质量调整推理策略
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

    # 判断信息质量和类型
    has_valid_consensus = (consensus_text and consensus_text.strip() and
                          "ConsensusMissingInInput" not in consensus_text and
                          not any(marker in consensus_text.lower() for marker in ["no consensus answer", "insufficient evidenc"]))

    has_valid_evidence = (additional_evidence_list and isinstance(additional_evidence_list, list) and
                         any(e and e.strip() and len(e.strip()) > 20 for e in additional_evidence_list))

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

    if has_valid_consensus or has_valid_evidence:
        # 先让模型基于内部知识思考
        prompt += "[No Retrieval]Let me first consider what I know about this question from my training data.\n\n"

        # 然后提供检索信息
        prompt += "[Retrieval]I also have access to retrieved information:\n"

        # 添加共识信息
        if has_valid_consensus:
            prompt += "<paragraph>{0}</paragraph>".format(consensus_text.strip())

        # 添加额外证据
        if has_valid_evidence:
            evidence_parts = []
            for evidence in additional_evidence_list[:2]:  # 最多2个证据
                if evidence and evidence.strip() and len(evidence.strip()) > 20:
                    evidence_parts.append(evidence.strip())

            if evidence_parts:
                prompt += "[Continue to Use Evidence]<paragraph>{0}</paragraph>".format(' '.join(evidence_parts))

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

def format_knowledge_first_prompt(item_index, task, query, consensus_text, additional_evidence_list, choices_data=None, item_choices=None):
    """
    知识优先格式：先让模型基于内部知识回答，再考虑检索信息
    """
    # 构建指令部分
    instruction = TASK_INST[task] + "\n\n## Input:\n\n" + query
    
    # 添加选项（对于ARC Challenge）
    if task == "arc_challenge":
        choices_to_use = get_arc_choices(item_choices, choices_data, item_index)
        instruction = format_arc_choices_for_instruction(choices_to_use, instruction)

    # 构建知识优先的prompt
    prompt = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)
    
    # 第一步：基于内部知识的初步判断
    prompt += ("First, let me consider what I know from my training data about this question.\n\n"
              "[No Retrieval]Based on my internal knowledge: ")
    
    # 判断是否有有效的检索信息
    has_valid_consensus = (consensus_text and consensus_text.strip() and 
                          "ConsensusMissingInInput" not in consensus_text and
                          not any(marker in consensus_text.lower() for marker in ["no consensus answer", "insufficient evidenc"]))
    
    has_valid_evidence = (additional_evidence_list and isinstance(additional_evidence_list, list) and
                         any(e and e.strip() and len(e.strip()) > 20 for e in additional_evidence_list))
    
    return prompt, has_valid_consensus, has_valid_evidence

def query_selfrag_adaptive(model, tokenizer, prompt, device, max_tokens=100, temperature=0.0):
    """
    自适应调用Self-RAG模型
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取新生成的部分
    prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False))
    answer = generated_text[prompt_length:].strip()
    
    return answer

def query_selfrag_two_stage(model, tokenizer, base_prompt, consensus_text, additional_evidence_list, 
                           device, max_tokens=100, temperature=0.0):
    """
    两阶段推理：先内部知识，再结合检索信息
    """
    # 第一阶段：基于内部知识
    stage1_answer = query_selfrag_adaptive(model, tokenizer, base_prompt, device, max_tokens//2, temperature)
    
    # 判断是否需要第二阶段
    confidence_indicators = ["I'm confident", "I know", "definitely", "certainly", "clearly"]
    uncertainty_indicators = ["I'm not sure", "I don't know", "unclear", "uncertain", "might be", "could be"]
    
    is_confident = any(indicator in stage1_answer.lower() for indicator in confidence_indicators)
    is_uncertain = any(indicator in stage1_answer.lower() for indicator in uncertainty_indicators)
    
    # 如果模型不确定，或者有高质量检索信息，进行第二阶段
    has_valid_info = ((consensus_text and len(consensus_text.strip()) > 50) or 
                     (additional_evidence_list and any(len(e.strip()) > 50 for e in additional_evidence_list if e)))
    
    if is_uncertain or (has_valid_info and not is_confident):
        # 第二阶段：结合检索信息
        stage2_prompt = base_prompt + stage1_answer + "\n\n"
        stage2_prompt += "Now let me also consider the retrieved information:\n\n[Retrieval]"
        
        # 添加检索信息
        if consensus_text and consensus_text.strip():
            stage2_prompt += f"<paragraph>Consensus: {consensus_text.strip()}</paragraph>"
        
        if additional_evidence_list:
            evidence_parts = [e.strip() for e in additional_evidence_list[:2] if e and e.strip()]
            if evidence_parts:
                stage2_prompt += f"[Continue to Use Evidence]<paragraph>{' '.join(evidence_parts)}</paragraph>"
        
        stage2_prompt += "\n\nBased on both my knowledge and the retrieved information, my final answer is: "
        
        # 生成最终答案
        final_answer = query_selfrag_adaptive(model, tokenizer, stage2_prompt, device, max_tokens//2, temperature)
        return stage1_answer + " " + final_answer
    else:
        # 模型很确定，直接返回第一阶段答案
        return stage1_answer

def postprocess_answer(answer, task):
    """
    改进的答案后处理
    """
    # 清理控制tokens
    for token in control_tokens:
        answer = answer.replace(token, "")
    answer = answer.replace("</s>", "").replace("\n", " ").replace("<|endoftext|>", "").strip()

    if task == "arc_challenge":
        return postprocess_arc_answer_unified(answer)
    elif task == "popqa":
        return postprocess_popqa_answer(answer)
    elif task == "pubqa":
        return postprocess_pubqa_answer(answer)
    
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--selfrag_model_path', type=str, required=True, help='Path to Self-RAG model')
    parser.add_argument('--task', type=str, required=True, choices=['popqa', 'arc_challenge', 'bio', 'pubqa'])
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for generation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--strategy', type=str, default='adaptive',
                       choices=['adaptive', 'enhanced_adaptive', 'two_stage'],
                       help='Reasoning strategy: adaptive, enhanced_adaptive, or two_stage')
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='Number of samples to process (-1 for all)')

    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading Self-RAG model from {args.selfrag_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.selfrag_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.selfrag_model_path,
        torch_dtype=torch.float16,
        device_map=None
    )
    model = model.to(args.device)
    
    # 设置特殊tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 处理数据
    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # 限制样本数量
    if args.num_samples > 0:
        data = data[:args.num_samples]

    results = []
    
    for i, item in enumerate(tqdm(data, desc=f"Processing with {args.strategy} strategy")):
        query = item['query']
        consensus_text = item.get('consensus', '')
        additional_evidence = item.get('additional_evidence', [])
        
        try:
            if args.strategy == 'adaptive':
                # 自适应策略
                prompt = format_adaptive_prompt(
                    i, args.task, query, consensus_text, additional_evidence,
                    item.get('choices_data'), item.get('choices')
                )
                answer = query_selfrag_adaptive(model, tokenizer, prompt, args.device,
                                              args.max_tokens, args.temperature)

            elif args.strategy == 'enhanced_adaptive':
                # 增强自适应策略：根据信息质量调整推理策略
                prompt = format_enhanced_adaptive_prompt(
                    i, args.task, query, consensus_text, additional_evidence,
                    item.get('choices_data'), item.get('choices')
                )
                answer = query_selfrag_adaptive(model, tokenizer, prompt, args.device,
                                              args.max_tokens, args.temperature)

            elif args.strategy == 'two_stage':
                # 两阶段策略
                base_prompt, has_consensus, has_evidence = format_knowledge_first_prompt(
                    i, args.task, query, consensus_text, additional_evidence,
                    item.get('choices_data'), item.get('choices')
                )
                answer = query_selfrag_two_stage(model, tokenizer, base_prompt, 
                                               consensus_text, additional_evidence,
                                               args.device, args.max_tokens, args.temperature)
            
            # 后处理答案
            processed_answer = postprocess_answer(answer, args.task)
            
            # 保存结果 - 使用与原始方法一致的字段名
            result = {
                'query': query,
                'processed_answer': processed_answer,  # 评估脚本期望的字段名
                'raw_selfrag_response': answer,        # 与原始方法一致
                'method': f'selfrag_{args.strategy}'
            }
            
            # 保留原始字段
            for key in ['id', 'answerKey', 'choices', 'choices_data']:
                if key in item:
                    result[key] = item[key]
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            # 添加错误结果
            results.append({
                'query': query,
                'processed_answer': '',
                'raw_selfrag_response': f'Error: {str(e)}',
                'method': f'selfrag_{args.strategy}'
            })
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
