"""
统一的ARC Challenge处理工具
用于确保所有推理脚本中ARC处理的一致性
"""

import re
import os
from typing import Optional, List, Dict, Union

def get_arc_choices(item_choices: Optional[Dict], choices_data: Optional[List[str]], item_index: int) -> Optional[Union[Dict, str]]:
    """
    统一的ARC选项获取逻辑
    
    Args:
        item_choices: 数据项中的选项信息
        choices_data: 外部选项数据列表
        item_index: 当前项的索引
    
    Returns:
        选项数据，可能是字典格式或字符串格式
    """
    # 优先使用数据项中的选项
    if item_choices:
        return item_choices
    
    # 然后使用外部选项数据
    if choices_data and item_index < len(choices_data):
        return choices_data[item_index]
    
    return None

def format_arc_choices_for_instruction(choices_to_use: Union[Dict, str], instruction: str) -> str:
    """
    统一的ARC选项格式化逻辑
    
    Args:
        choices_to_use: 选项数据
        instruction: 原始指令
    
    Returns:
        格式化后的指令（包含选项）
    """
    if not choices_to_use:
        return instruction
    
    if isinstance(choices_to_use, dict) and 'text' in choices_to_use and 'label' in choices_to_use:
        # 新格式: {"text": [...], "label": ["A", "B", "C", "D"]}
        choices_formatted = ""
        for label, text in zip(choices_to_use['label'], choices_to_use['text']):
            choices_formatted += f"\n{label}: {text}"
        return instruction + choices_formatted
    
    elif isinstance(choices_to_use, str):
        # 旧格式: 字符串格式
        from core.utils import format_arc_choices_for_prompt
        choices_text = format_arc_choices_for_prompt(choices_to_use)
        if choices_text:
            choices_formatted = choices_text.replace("A)", "\nA: ").replace("B)", "\nB: ").replace("C)", "\nC: ").replace("D)", "\nD: ")
            return instruction + choices_formatted
    
    return instruction

def find_arc_choices_file(input_file: str) -> Optional[str]:
    """
    统一的ARC选项文件查找逻辑
    
    Args:
        input_file: 输入文件路径
    
    Returns:
        找到的选项文件路径，如果没找到则返回None
    """
    # 定义所有可能的文件名模式
    possible_patterns = [
        '_enhanced_consensus_evidence_v3_fixed.jsonl',
        '_enhanced_consensus_evidence.jsonl',
        '_full_enhanced_consensus_evidence.jsonl',
        '_enhanced_consensus_evidence_v2.jsonl',
        '_enhanced_consensus_evidence_v1.jsonl'
    ]
    
    possible_choices_files = []
    
    # 为每个模式生成可能的路径
    for pattern in possible_patterns:
        if pattern in input_file:
            # 替换为choices文件
            choices_file = input_file.replace(pattern, '_choices.txt')
            possible_choices_files.append(choices_file)
            
            # 尝试从enhanced_eval目录映射到原始数据目录
            if 'enhanced_eval/' in choices_file:
                original_path = choices_file.replace('enhanced_eval/', 'arc_challenge/splits/')
                possible_choices_files.append(original_path)
    
    # 添加默认路径
    possible_choices_files.extend([
        '/workspace/conRAG/data/arc_challenge/splits/arc_challenge_eval_scientific_choices.txt',
        '/workspace/conRAG/data/arc_challenge/splits/arc_challenge_full_choices.txt',
        '/workspace/conRAG/data/arc_challenge/splits/arc_challenge_test_choices.txt'
    ])
    
    # 查找第一个存在的文件
    for choices_file in possible_choices_files:
        if os.path.exists(choices_file):
            return choices_file
    
    return None

def load_arc_choices_data(choices_file: str) -> List[str]:
    """
    加载ARC选项数据
    
    Args:
        choices_file: 选项文件路径
    
    Returns:
        选项数据列表
    """
    try:
        with open(choices_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Error loading choices file {choices_file}: {e}")
        return []

def postprocess_arc_answer_unified(answer: str, original_answer: str = None) -> str:
    """
    统一的ARC答案后处理逻辑
    
    Args:
        answer: 处理后的答案
        original_answer: 原始答案（用于fallback）
    
    Returns:
        最终的答案字母 (A, B, C, D)
    """
    if not answer:
        return "A"
    
    answer = answer.strip()
    
    # 1. 查找明确的答案模式
    answer_patterns = [
        r'(?:the\s+)?answer\s+is:?\s*([A-D])',
        r'(?:correct\s+)?(?:answer|choice)\s+is:?\s*([A-D])',
        r'(?:I\s+)?(?:choose|select)\s+([A-D])',
        r'(?:option|choice)\s+([A-D])',
        r'^([A-D])\.?\s*$',  # 只有单个字母
        r'^([A-D])[:,\s]',   # 字母开头
        r'([A-D])\s*is\s+(?:correct|right|the\s+answer)',
        r'([A-D])\s*is\s+the\s+(?:best|correct)\s+(?:answer|choice)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, answer, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    # 2. 使用原有的后处理函数
    try:
        from core.utils import postprocess_arc_answer
        processed = postprocess_arc_answer(answer)
        if processed in ["A", "B", "C", "D"]:
            return processed
    except:
        pass
    
    # 3. 在原始答案中查找（如果提供）
    if original_answer:
        for pattern in answer_patterns:
            match = re.search(pattern, original_answer, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
    
    # 4. 查找答案中的第一个A-D字母
    first_letter = re.search(r'[A-D]', answer, re.IGNORECASE)
    if first_letter:
        return first_letter.group(0).upper()
    
    # 5. 最后的fallback
    if answer and answer[0].upper() in "ABCD":
        return answer[0].upper()
    
    return "A"  # 默认值

def setup_arc_processing(input_file: str, task: str) -> Optional[List[str]]:
    """
    统一的ARC处理设置
    
    Args:
        input_file: 输入文件路径
        task: 任务类型
    
    Returns:
        选项数据列表，如果不是ARC任务或找不到选项文件则返回None
    """
    if task != "arc_challenge":
        return None
    
    choices_file = find_arc_choices_file(input_file)
    if choices_file:
        choices_data = load_arc_choices_data(choices_file)
        print(f"Found ARC choices file: {choices_file} ({len(choices_data)} choices)")
        return choices_data
    else:
        print(f"Warning: No ARC choices file found for input: {input_file}")
        return None

# 使用示例和测试函数
def test_arc_utils():
    """测试ARC工具函数"""
    # 测试答案后处理
    test_cases = [
        ("The answer is A", "A"),
        ("I choose B.", "B"),
        ("Option C is correct", "C"),
        ("D", "D"),
        ("The correct answer is option A.", "A"),
        ("Based on the analysis, B is the best choice.", "B"),
        ("", "A"),  # 默认值
    ]
    
    print("Testing ARC answer postprocessing:")
    for input_text, expected in test_cases:
        result = postprocess_arc_answer_unified(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_text}' -> '{result}' (expected: '{expected}')")

if __name__ == "__main__":
    test_arc_utils()
