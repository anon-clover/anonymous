# utils.py
import re
from typing import Dict, List, Any

def format_arc_choices_for_prompt(choices_data: Dict) -> str:
    """
    格式化选项用于prompt，每个选项换行
    输入: {"text": [...], "label": [...]}
    输出: "\nA: text1\nB: text2\nC: text3\nD: text4" 或 "\n1: text1\n2: text2\n3: text3\n4: text4"
    """
    if not choices_data:
        return ""
    
    if isinstance(choices_data, dict) and 'text' in choices_data and 'label' in choices_data:
        formatted = ""
        for label, text in zip(choices_data['label'], choices_data['text']):
            formatted += f"\n{label}: {text}"
        return formatted
    
    return ""

def postprocess_arc_answer(answer: str) -> str:
    """
    后处理ARC答案，提取选项字母或数字
    """
    # 清理答案
    answer = answer.strip()
    
    # 尝试多种模式提取答案（包括ABCD和1234）
    patterns = [
        r'^([A-D1-4])(?:[.)\s:]|$)',  # 开头的A. 或 A) 或 A: 或 就是A，也支持1-4
        r'answer is ([A-D1-4])',
        r'correct answer is ([A-D1-4])',
        r'choose ([A-D1-4])',
        r'option ([A-D1-4])',
        r'\b([A-D1-4])\b'  # 独立的字母或数字
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            result = match.group(1).upper()
            # 如果是数字，直接返回；如果是字母，返回大写
            return result
    
    # 如果答案就是单个字母或数字
    if len(answer) == 1:
        if answer.upper() in "ABCD":
            return answer.upper()
        elif answer in "1234":
            return answer
    
    # 返回原始答案的前10个字符（用于调试）
    return answer[:10] if len(answer) > 10 else answer

def postprocess_popqa_answer(answer: str) -> str:
    """
    后处理POPQA答案，提取核心答案部分
    去除解释性文本，只保留实际答案
    """
    if not answer:
        return answer
    
    # 清理答案
    answer = answer.strip()
    
    # 移除常见的前缀短语
    prefixes_to_remove = [
        "based on the provided context",
        "according to the context",
        "the answer is:",
        "answer:"
    ]
    
    lower_answer = answer.lower()
    for prefix in prefixes_to_remove:
        if lower_answer.startswith(prefix):
            # 找到前缀后的内容
            start_idx = len(prefix)
            # 跳过冒号、逗号等标点
            while start_idx < len(answer) and answer[start_idx] in ' :,':
                start_idx += 1
            answer = answer[start_idx:].strip()
            break
    
    # 对于较短的答案（少于20个字符），保留原样
    if len(answer) < 20:
        return answer
    
    # 如果答案包含多个句子，保留前两句
    sentences = answer.split('. ')
    if len(sentences) > 2:
        answer = '. '.join(sentences[:2]) + '.'
    
    # 移除括号中的额外信息（仅当答案较长时）
    if len(answer) > 50 and '(' in answer:
        parts = answer.split('(')
        if len(parts[0].strip()) > 20:  # 确保主要答案足够完整
            answer = parts[0].strip()
    
    return answer

def postprocess_pubqa_answer(answer):
    """
    后处理PubQA任务的答案，确保输出为true或false

    Args:
        answer (str): 模型生成的原始答案

    Returns:
        str: 处理后的答案，只能是'true'或'false'
    """
    if not answer:
        return "false"

    # 转换为小写并清理
    answer = answer.lower().strip()

    # 移除常见的前缀和后缀
    prefixes_to_remove = [
        "answer:", "response:", "the answer is", "the statement is",
        "this is", "it is", "the correct answer is", "my answer is"
    ]

    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    # 移除标点符号
    answer = answer.strip('.,!?;:"\'()[]{}')

    # 检查true的各种表达
    true_indicators = [
        'true', 'yes', 'correct', 'right', 'accurate', 'valid',
        'confirmed', 'supported', 'verified', '1', 'positive'
    ]

    # 检查false的各种表达
    false_indicators = [
        'false', 'no', 'incorrect', 'wrong', 'inaccurate', 'invalid',
        'not correct', 'not true', 'not right', 'unsupported', '0', 'negative'
    ]

    # 精确匹配
    if answer in true_indicators:
        return "true"
    elif answer in false_indicators:
        return "false"

    # 部分匹配（包含关键词）
    for indicator in true_indicators:
        if indicator in answer:
            return "true"

    for indicator in false_indicators:
        if indicator in answer:
            return "false"

    # 如果都没有匹配，默认返回false（保守策略）
    return "false"