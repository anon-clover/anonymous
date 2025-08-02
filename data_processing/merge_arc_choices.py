# merge_arc_choices.py (已修复)
import json
import argparse
from tqdm import tqdm

def ensure_arc_data_has_choices(enhanced_file, original_file_with_choices, output_file):
    """
    为从 `enhanced_file` 读取的数据合并 `original_file_with_choices` 中的选项和答案，
    然后将结果写入 `output_file`。
    """
    print(f"Reading original data with choices from: {original_file_with_choices}")
    choices_map = {}
    with open(original_file_with_choices, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data.get('question', '').strip()
            if question:
                choices_map[question] = {
                    'choices': data.get('choices'),
                    'answerKey': data.get('answerKey')
                }
    
    print(f"Loaded {len(choices_map)} questions with choices.")
    
    print(f"Reading enhanced data from: {enhanced_file}")
    updated_data = []
    matched_count = 0
    
    with open(enhanced_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Merging choices"):
            data = json.loads(line)
            query = data.get('query', '').strip()
            
            # 尝试精确匹配
            if query in choices_map:
                data['choices'] = choices_map[query]['choices']
                data['answerKey'] = choices_map[query]['answerKey']
                matched_count += 1
            else:
                # 如果精确匹配失败，可以尝试进行模糊匹配（可选）
                found_fuzzy = False
                for orig_q, choice_info in choices_map.items():
                    if query.lower() in orig_q.lower() or orig_q.lower() in query.lower():
                        data['choices'] = choice_info['choices']
                        data['answerKey'] = choice_info['answerKey']
                        matched_count += 1
                        found_fuzzy = True
                        break
                if not found_fuzzy:
                     # 即使找不到匹配，也保留原始数据，但标记一下
                    data['choices'] = None
                    data['answerKey'] = None

            updated_data.append(data)
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Matched {matched_count}/{len(updated_data)} items with choices.")
    print(f"Output saved to: {output_file}")


def main():
    """主函数，用于解析命令行参数。"""
    parser = argparse.ArgumentParser(description="为ARC数据集的增强文件合并选项和答案。")
    parser.add_argument('--enhanced_file', type=str, required=True, help='需要被处理的输入文件路径 (例如，包含共识的文件)。')
    parser.add_argument('--original_file_with_choices', type=str, required=True, help='包含选项和答案的标准答案文件路径。')
    parser.add_argument('--output_file', type=str, required=True, help='合并后用于推理的最终输出文件路径。')
    args = parser.parse_args()
    
    ensure_arc_data_has_choices(
        args.enhanced_file,
        args.original_file_with_choices,
        args.output_file
    )

if __name__ == "__main__":
    main()
