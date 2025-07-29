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

def format_prompt(item_index, task, query, consensus_text, additional_evidence_list, choices_data=None, item_choices=None, original_passages=None):
    """
    使用Self-RAG的标准格式，更好地利用共识和额外证据
    """
    # 构建指令部分
    instruction = TASK_INST[task] + "\n\n## Input:\n\n" + query
    
    # 添加选项（对于ARC Challenge）- 使用统一的处理逻辑
    if task == "arc_challenge":
        choices_to_use = get_arc_choices(item_choices, choices_data, item_index)
        instruction = format_arc_choices_for_instruction(choices_to_use, instruction)

    # 构建Self-RAG格式的prompt
    prompt = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)
    
    # 判断是否有有效的证据
    has_valid_consensus = (consensus_text and consensus_text.strip() and 
                          "ConsensusMissingInInput" not in consensus_text and
                          not any(marker in consensus_text.lower() for marker in ["no consensus answer", "insufficient evidenc"]))
    
    has_valid_evidence = (additional_evidence_list and isinstance(additional_evidence_list, list) and
                         any(e and e.strip() and len(e.strip()) > 20 for e in additional_evidence_list))
    
    if has_valid_consensus or has_valid_evidence:
        # 使用[Retrieval]标记表示使用外部知识
        prompt += "[Retrieval]"
        
        # 1. 优先添加共识（通常是最相关的信息）
        if has_valid_consensus:
            prompt += "<paragraph>Consensus: {0}</paragraph>".format(consensus_text.strip())
        
        # 2. 添加额外证据
        if has_valid_evidence:
            # 过滤并选择最相关的证据
            evidence_parts = []
            for evidence in additional_evidence_list[:3]:  # 最多3个证据
                if evidence and evidence.strip() and len(evidence.strip()) > 20:
                    evidence_parts.append(evidence.strip())
            
            if evidence_parts:
                # 用[Continue to Use Evidence]标记额外证据
                evidence_combined = ' '.join(evidence_parts)
                prompt += "[Continue to Use Evidence]<paragraph>{0}</paragraph>".format(evidence_combined)
    else:
        # 没有有效证据时，使用[No Retrieval]
        prompt += "[No Retrieval]"
    
    return prompt

def postprocess_answer(answer, task):
    """
    改进的答案后处理，更准确地提取答案
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

def query_selfrag(model, tokenizer, prompt, device, max_tokens=100, temperature=0.0):
    """
    调用Self-RAG模型生成答案
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,  # 避免0值错误
            top_p=1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
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
        consensus_text = item.get('consensus', '')
        additional_evidence = item.get('additional_evidence', [])
        item_choices = item.get('choices', None)
        original_passages = item.get('original_passages', [])
        
        prompt = format_prompt(i, args.task, query, consensus_text, additional_evidence, 
                             choices_data, item_choices, original_passages)
        raw_answer = query_selfrag(model, tokenizer, prompt, device, args.max_tokens, args.temperature)
        final_answer = postprocess_answer(raw_answer, args.task)
        
        results.append({
            "query": query,
            "raw_selfrag_response": raw_answer,
            "processed_answer": final_answer,
            "method": "selfrag_enhanced"
        })
    
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"SelfRAG Enhanced (共识+证据)完成处理 {len(results)} 个样本")

if __name__ == "__main__":
    main()