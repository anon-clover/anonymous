#!/usr/bin/env python3
"""
统一推理策略配置
为Self-RAG和Ollama模型提供一致的推理逻辑，但保持各自的prompt风格
"""

from core.arc_utils import get_arc_choices, format_arc_choices_for_instruction

TASK_INST = {
    "popqa": "Answer the following question based on your knowledge and any provided information.",
    "arc_challenge": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "bio": "Generate a comprehensive biography based on your knowledge and any provided information.",
    "pubqa": "Is the following statement correct or not? Say true if it's correct; otherwise say false."
}

class ReasoningStrategy:
    """统一的推理策略基类"""
    
    @staticmethod
    def analyze_information_quality(consensus_text, additional_evidence):
        """
        统一的信息质量分析逻辑
        返回信息质量指标，供不同模型使用
        """
        # 基础有效性检查
        has_valid_consensus = (consensus_text and consensus_text.strip() and 
                              "ConsensusMissingInInput" not in consensus_text and
                              not any(marker in consensus_text.lower() for marker in ["no consensus answer", "insufficient evidenc"]))
        
        has_valid_evidence = (additional_evidence and isinstance(additional_evidence, list) and
                             any(e and e.strip() and len(e.strip()) > 20 for e in additional_evidence))
        
        # 深度质量分析
        consensus_lower = consensus_text.lower() if consensus_text else ""
        is_information_insufficient = any(phrase in consensus_lower for phrase in 
                                        ["do not contain", "do not directly address", "insufficient", 
                                         "no information", "not mentioned", "does not provide"])
        
        has_clear_support = any(word in consensus_lower for word in 
                               ['support', 'confirm', 'show', 'demonstrate', 'indicate', 'found', 'study'])
        has_clear_refute = any(word in consensus_lower for word in 
                              ['contradict', 'refute', 'oppose', 'against', 'deny'])
        
        return {
            'has_valid_consensus': has_valid_consensus,
            'has_valid_evidence': has_valid_evidence,
            'is_information_insufficient': is_information_insufficient,
            'has_clear_support': has_clear_support,
            'has_clear_refute': has_clear_refute,
            'consensus_lower': consensus_lower
        }
    
    @staticmethod
    def get_reasoning_guidance(quality_info):
        """
        根据信息质量提供推理指导
        """
        if quality_info['is_information_insufficient']:
            return "insufficient_info"
        elif quality_info['has_clear_support'] or quality_info['has_clear_refute']:
            return "clear_evidence"
        else:
            return "balanced_reasoning"

class SelfRAGStrategy(ReasoningStrategy):
    """Self-RAG模型的推理策略"""
    
    @staticmethod
    def format_enhanced_adaptive_prompt(item_index, task, query, consensus_text, additional_evidence_list, choices_data=None, item_choices=None):
        """
        Self-RAG增强自适应推理格式
        使用统一的推理逻辑，但保持Self-RAG的token风格
        """
        # 构建指令部分
        instruction = TASK_INST[task] + "\n\n## Input:\n\n" + query
        
        # 添加选项（对于ARC Challenge）
        if task == "arc_challenge":
            choices_to_use = get_arc_choices(item_choices, choices_data, item_index)
            instruction = format_arc_choices_for_instruction(choices_to_use, instruction)

        prompt = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)
        
        # 使用统一的信息质量分析
        quality_info = SelfRAGStrategy.analyze_information_quality(consensus_text, additional_evidence_list)
        reasoning_guidance = SelfRAGStrategy.get_reasoning_guidance(quality_info)
        
        if quality_info['has_valid_consensus'] or quality_info['has_valid_evidence']:
            # 先让模型基于内部知识思考
            prompt += "[No Retrieval]Let me first consider what I know about this question from my training data.\n\n"

            # 然后提供检索信息
            prompt += "[Retrieval]I also have access to retrieved information:\n"
            prompt += "<paragraph>{0}</paragraph>".format(consensus_text.strip())

            # 添加额外证据
            if quality_info['has_valid_evidence']:
                evidence_parts = []
                for evidence in additional_evidence_list[:3]:
                    if evidence and evidence.strip() and len(evidence.strip()) > 20:
                        evidence_parts.append(evidence.strip())

                if evidence_parts:
                    prompt += "\n[Continue to Use Evidence]Additional supporting evidence: {0}".format(
                        " | ".join(evidence_parts))

            # 根据推理指导调整策略
            if reasoning_guidance == "insufficient_info":
                prompt += "\n\nThe retrieved information appears to be insufficient. I should rely more on my training knowledge to answer this question. "
            elif reasoning_guidance == "clear_evidence":
                prompt += "\n\nThe retrieved information provides clear evidence. I should prioritize this evidence in my reasoning. "
            else:
                prompt += "\n\nBased on both my knowledge and the retrieved information, "

        else:
            # 没有检索信息时，直接使用内部知识
            prompt += "[No Retrieval]"

        return prompt

class OllamaIntelligentStrategy(ReasoningStrategy):
    """Ollama智能RAG的推理策略"""
    
    @staticmethod
    def format_selfrag_inspired_adaptive_prompt(item_index, task, query, consensus_text, additional_evidence, choices_data=None, item_choices=None):
        """
        Ollama智能RAG格式
        使用统一的推理逻辑，但保持结构化步骤风格
        """
        # 构建指令部分
        instruction = TASK_INST[task] + "\n\n## Input:\n\n" + query

        # 添加选项（对于ARC Challenge）
        if task == "arc_challenge":
            choices_to_use = get_arc_choices(item_choices, choices_data, item_index)
            instruction = format_arc_choices_for_instruction(choices_to_use, instruction)

        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        # 使用统一的信息质量分析
        quality_info = OllamaIntelligentStrategy.analyze_information_quality(consensus_text, additional_evidence)
        reasoning_guidance = OllamaIntelligentStrategy.get_reasoning_guidance(quality_info)

        if quality_info['has_valid_consensus'] or quality_info['has_valid_evidence']:
            # 阶段1：内部知识思考
            prompt += "**Step 1 - Internal Knowledge:**\n"
            prompt += "Let me first consider what I know from my training data.\n\n"

            # 阶段2：检索信息呈现
            prompt += "**Step 2 - Retrieved Information:**\n"

            # 添加共识信息
            if quality_info['has_valid_consensus']:
                prompt += f"Consensus: {consensus_text.strip()}\n\n"

            # 添加额外证据
            if quality_info['has_valid_evidence']:
                evidence_parts = []
                for evidence in additional_evidence[:2]:
                    if evidence and evidence.strip() and len(evidence.strip()) > 20:
                        evidence_parts.append(evidence.strip())

                if evidence_parts:
                    prompt += f"Additional evidence: {' '.join(evidence_parts)}\n\n"

            # 阶段3：智能融合策略
            prompt += "**Step 3 - Intelligent Integration:**\n"
            
            if reasoning_guidance == "insufficient_info":
                prompt += "The retrieved information seems insufficient. I'll rely more on my training knowledge while noting the limitations.\n\n"
            elif reasoning_guidance == "clear_evidence":
                prompt += "The retrieved information provides clear evidence. I'll prioritize this evidence in my reasoning.\n\n"
            else:
                prompt += "I'll carefully integrate both my training knowledge and the retrieved information.\n\n"

            prompt += "**Final Answer:** "

        else:
            # 没有检索信息时
            prompt += "**Internal Knowledge Analysis:**\nNo retrieved info. Using training knowledge. "

        return prompt

class OllamaConsensusStrategy(ReasoningStrategy):
    """Ollama完整共识RAG的推理策略"""
    
    @staticmethod
    def format_enhanced_adaptive_prompt_llama(item_index, task, query, consensus_text, additional_evidence, choices_data=None, item_choices=None):
        """
        Ollama完整共识RAG格式
        使用统一的推理逻辑，但保持简化的描述性风格
        """
        # 构建指令部分
        instruction = TASK_INST[task] + "\n\n## Input:\n\n" + query

        # 添加选项（对于ARC Challenge）
        if task == "arc_challenge":
            choices_to_use = get_arc_choices(item_choices, choices_data, item_index)
            instruction = format_arc_choices_for_instruction(choices_to_use, instruction)

        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        # 使用统一的信息质量分析
        quality_info = OllamaConsensusStrategy.analyze_information_quality(consensus_text, additional_evidence)
        reasoning_guidance = OllamaConsensusStrategy.get_reasoning_guidance(quality_info)

        if quality_info['has_valid_consensus'] or quality_info['has_valid_evidence']:
            # 先基于内部知识思考
            prompt += "Let me first consider what I know about this question from my training data.\n\n"

            # 然后提供检索信息
            prompt += "I also have access to retrieved information:\n\n"

            # 添加共识信息
            if quality_info['has_valid_consensus']:
                prompt += f"**Consensus Analysis:**\n{consensus_text.strip()}\n\n"

            # 添加额外证据
            if quality_info['has_valid_evidence']:
                evidence_parts = []
                for evidence in additional_evidence[:2]:
                    if evidence and evidence.strip() and len(evidence.strip()) > 20:
                        evidence_parts.append(evidence.strip())

                if evidence_parts:
                    prompt += f"**Additional Evidence:**\n{' '.join(evidence_parts)}\n\n"

            # 根据推理指导调整策略
            if reasoning_guidance == "insufficient_info":
                prompt += "The retrieved information appears to be insufficient. I should rely more on my training knowledge to answer this question. "
            elif reasoning_guidance == "clear_evidence":
                prompt += "The retrieved information provides clear evidence. I should prioritize this evidence in my reasoning. "
            else:
                prompt += "Based on both my knowledge and the retrieved information, "

        else:
            # 没有检索信息时
            prompt += "I don't have specific retrieved information for this question, so I'll rely on my training knowledge. "

        return prompt

# 策略映射
STRATEGY_MAP = {
    'selfrag': SelfRAGStrategy,
    'ollama_intelligent': OllamaIntelligentStrategy,
    'ollama_consensus': OllamaConsensusStrategy
}
