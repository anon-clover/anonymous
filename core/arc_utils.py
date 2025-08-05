

import re
import os
from typing import Optional, List, Dict, Union

def get_arc_choices(item_choices: Optional[Dict], choices_data: Optional[List[str]], item_index: int) -> Optional[Union[Dict, str]]:

    if item_choices:
        return item_choices
    

    if choices_data and item_index < len(choices_data):
        return choices_data[item_index]
    
    return None

def format_arc_choices_for_instruction(choices_to_use: Union[Dict, str], instruction: str) -> str:

    if not choices_to_use:
        return instruction
    
    if isinstance(choices_to_use, dict) and 'text' in choices_to_use and 'label' in choices_to_use:
        # 新格式: {"text": [...], "label": ["A", "B", "C", "D"]}
        choices_formatted = ""
        for label, text in zip(choices_to_use['label'], choices_to_use['text']):
            choices_formatted += f"\n{label}: {text}"
        return instruction + choices_formatted
    
    elif isinstance(choices_to_use, str):

        from core.utils import format_arc_choices_for_prompt
        choices_text = format_arc_choices_for_prompt(choices_to_use)
        if choices_text:
            choices_formatted = choices_text.replace("A)", "\nA: ").replace("B)", "\nB: ").replace("C)", "\nC: ").replace("D)", "\nD: ")
            return instruction + choices_formatted
    
    return instruction

def find_arc_choices_file(input_file: str) -> Optional[str]:

    possible_patterns = [
        '_enhanced_consensus_evidence_v3_fixed.jsonl',
        '_enhanced_consensus_evidence.jsonl',
        '_full_enhanced_consensus_evidence.jsonl',
        '_enhanced_consensus_evidence_v2.jsonl',
        '_enhanced_consensus_evidence_v1.jsonl'
    ]
    
    possible_choices_files = []
    
    for pattern in possible_patterns:
        if pattern in input_file:
            choices_file = input_file.replace(pattern, '_choices.txt')
            possible_choices_files.append(choices_file)
            
            if 'enhanced_eval/' in choices_file:
                original_path = choices_file.replace('enhanced_eval/', 'arc_challenge/splits/')
                possible_choices_files.append(original_path)
    
    possible_choices_files.extend([
        'data/arc_challenge/splits/arc_challenge_eval_scientific_choices.txt',
        'data/arc_challenge/splits/arc_challenge_full_choices.txt',
        'data/arc_challenge/splits/arc_challenge_test_choices.txt'
    ])
    

    for choices_file in possible_choices_files:
        if os.path.exists(choices_file):
            return choices_file
    
    return None

def load_arc_choices_data(choices_file: str) -> List[str]:

    try:
        with open(choices_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Error loading choices file {choices_file}: {e}")
        return []

def postprocess_arc_answer_unified(answer: str, original_answer: str = None) -> str:

    if not answer:
        return "A"
    
    answer = answer.strip()

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
    
    try:
        from core.utils import postprocess_arc_answer
        processed = postprocess_arc_answer(answer)
        if processed in ["A", "B", "C", "D"]:
            return processed
    except:
        pass

    if original_answer:
        for pattern in answer_patterns:
            match = re.search(pattern, original_answer, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
    

    first_letter = re.search(r'[A-D]', answer, re.IGNORECASE)
    if first_letter:
        return first_letter.group(0).upper()

    if answer and answer[0].upper() in "ABCD":
        return answer[0].upper()
    
    return "A"  # 默认值

def setup_arc_processing(input_file: str, task: str) -> Optional[List[str]]:

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


def test_arc_utils():

    test_cases = [
        ("The answer is A", "A"),
        ("I choose B.", "B"),
        ("Option C is correct", "C"),
        ("D", "D"),
        ("The correct answer is option A.", "A"),
        ("Based on the analysis, B is the best choice.", "B"),
        ("", "A"),  
    ]
    
    print("Testing ARC answer postprocessing:")
    for input_text, expected in test_cases:
        result = postprocess_arc_answer_unified(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_text}' -> '{result}' (expected: '{expected}')")

if __name__ == "__main__":
    test_arc_utils()
