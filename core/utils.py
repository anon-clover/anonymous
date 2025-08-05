# utils.py
import re
from typing import Dict, List, Any

def format_arc_choices_for_prompt(choices_data: Dict) -> str:

    if not choices_data:
        return ""
    
    if isinstance(choices_data, dict) and 'text' in choices_data and 'label' in choices_data:
        formatted = ""
        for label, text in zip(choices_data['label'], choices_data['text']):
            formatted += f"\n{label}: {text}"
        return formatted
    
    return ""

def postprocess_arc_answer(answer: str) -> str:

    answer = answer.strip()

    patterns = [
        r'^([A-D1-4])(?:[.)\s:]|$)', 
        r'answer is ([A-D1-4])',
        r'correct answer is ([A-D1-4])',
        r'choose ([A-D1-4])',
        r'option ([A-D1-4])',
        r'\b([A-D1-4])\b'  
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            result = match.group(1).upper()
            return result
    

    if len(answer) == 1:
        if answer.upper() in "ABCD":
            return answer.upper()
        elif answer in "1234":
            return answer
    
    return answer[:10] if len(answer) > 10 else answer

def postprocess_popqa_answer(answer: str) -> str:

    if not answer:
        return answer
    

    answer = answer.strip()
    

    prefixes_to_remove = [
        "based on the provided context",
        "according to the context",
        "the answer is:",
        "answer:"
    ]
    
    lower_answer = answer.lower()
    for prefix in prefixes_to_remove:
        if lower_answer.startswith(prefix):

            start_idx = len(prefix)

            while start_idx < len(answer) and answer[start_idx] in ' :,':
                start_idx += 1
            answer = answer[start_idx:].strip()
            break
    

    if len(answer) < 20:
        return answer
    

    sentences = answer.split('. ')
    if len(sentences) > 2:
        answer = '. '.join(sentences[:2]) + '.'
    

    if len(answer) > 50 and '(' in answer:
        parts = answer.split('(')
        if len(parts[0].strip()) > 20: 
            answer = parts[0].strip()
    
    return answer

def postprocess_pubqa_answer(answer):

    if not answer:
        return "false"


    answer = answer.lower().strip()


    prefixes_to_remove = [
        "answer:", "response:", "the answer is", "the statement is",
        "this is", "it is", "the correct answer is", "my answer is"
    ]

    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()


    answer = answer.strip('.,!?;:"\'()[]{}')


    true_indicators = [
        'true', 'yes', 'correct', 'right', 'accurate', 'valid',
        'confirmed', 'supported', 'verified', '1', 'positive'
    ]


    false_indicators = [
        'false', 'no', 'incorrect', 'wrong', 'inaccurate', 'invalid',
        'not correct', 'not true', 'not right', 'unsupported', '0', 'negative'
    ]


    if answer in true_indicators:
        return "true"
    elif answer in false_indicators:
        return "false"


    for indicator in true_indicators:
        if indicator in answer:
            return "true"

    for indicator in false_indicators:
        if indicator in answer:
            return "false"

    return "false"
