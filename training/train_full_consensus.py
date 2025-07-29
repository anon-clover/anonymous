import argparse
import os
import re
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_scheduler
import json

def clean_and_split_document(doc_str):
    """
    直接处理并分割 documents 字符串，支持混合的单引号和双引号
    """
    doc_str = doc_str.replace("'", '"')
    documents = re.findall(r'"(.*?)"', doc_str)
    return documents

def get_data(file):
    queries = []
    documents = []
    consensuses = []
    with open(file, "r", encoding="utf-8") as f:
        query = None
        document_list_str = None
        consensus_lines = []
        for line in f:
            line = line.strip()
            if line.startswith("query:"):
                if query and document_list_str and consensus_lines:
                    consensuses.append(" ".join(consensus_lines))
                query = line.split("query:")[1].strip()
                queries.append(query)
                consensus_lines = []
                document_list_str = None
            elif line.startswith("documents:"):
                document_list_str = line.split("documents:")[1].strip()
                try:
                    parsed_documents = json.loads(document_list_str)
                    if isinstance(parsed_documents, list):
                        documents.append([str(doc) for doc in parsed_documents])
                    else:
                        print(f"Warning: Documents format not a list for query '{query}': {document_list_str}")
                        documents.append([])
                except json.JSONDecodeError:
                    print(f"Warning: JSONDecodeError for documents: {document_list_str}. Trying regex.")
                    parsed_documents_regex = clean_and_split_document(document_list_str)
                    documents.append(parsed_documents_regex)

            elif line.startswith("consensus:"):
                consensus_lines.append(line.split("consensus:")[1].strip())
            else:
                if line and consensus_lines:
                    consensus_lines.append(line)
        
        # 最后一条记录的共识处理
        if query and document_list_str and consensus_lines:
             consensuses.append(" ".join(consensus_lines))
        
        # 确保数据长度一致
        if len(documents) < len(queries):
            documents.extend([[] for _ in range(len(queries) - len(documents))])
        if len(consensuses) < len(queries):
            consensuses.extend(["" for _ in range(len(queries) - len(consensuses))])

    return queries, documents, consensuses

def data_preprocess(file, tokenizer):
    queries, documents, consensuses = get_data(file)
    
    # 检查数据长度匹配
    if not (len(queries) == len(documents) == len(consensuses)):
        raise ValueError(f"Data length mismatch after get_data: "
                         f"Queries: {len(queries)}, Documents: {len(documents)}, Consensuses: {len(consensuses)}")

    # 为T5输入格式化文本
    input_texts = [f"query: {q} documents: {' '.join(d_list)}" for q, d_list in zip(queries, documents)]
    
    data = pd.DataFrame({"input_text": input_texts, "target_text": consensuses})
    
    print(f"Number of records for training: {len(data)}")
    if len(data) == 0:
        raise ValueError("No data loaded for training. Please check the input file and its parsing in get_data.")

    train_data = tokenizer(data.input_text.to_list(), padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    target_data = tokenizer(data.target_text.to_list(), padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    return train_data, target_data

def main():
    parser = argparse.ArgumentParser(description="Train T5 consensus model on full datasets")
    parser.add_argument('--train_file', type=str, 
                       default='/root/shh/FGRAG/fgrag/data/consensus_training/full_mixed_consensus_for_t5_training_gpt4o_2024_11_20.txt',
                       help='Training data file path')
    parser.add_argument('--save_path', type=str, 
                       default='/root/shh/FGRAG/fgrag/models/t5_full_consensus_gpt4o_2024_11_20',
                       help='Model save path')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs (default: 100, saves models only for last 10 epochs)')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    train_file = args.train_file
    SEED = args.seed

    # 设置随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED) 

    # 模型和分词器路径
    tokenizer_path = "/root/shh/FGRAG/fgrag/google-t5/t5-large"
    model_path = "/root/shh/FGRAG/fgrag/google-t5/t5-large"
    
    print(f"=== Full Dataset Consensus Model Training ===")
    print(f"Training file: {train_file}")
    print(f"Save path: {args.save_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    print(f"Loading model from: {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    print(f"Preprocessing data from: {train_file}")
    train_data, target_data = data_preprocess(train_file, tokenizer)

    # 配置训练
    batch_size = args.batch_size
    train_dataset = TensorDataset(train_data["input_ids"], train_data["attention_mask"], target_data["input_ids"])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # GPU配置
    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        if num_gpus > 1:
            print(f"Using DataParallel for {num_gpus} GPUs.")
            device_ids = list(range(num_gpus))
            model = nn.DataParallel(model, device_ids=device_ids)
        elif num_gpus == 1:
            print("Running on a single GPU. DataParallel not used.")
        else:
             print("No GPUs available. Running on CPU. DataParallel not used.")
    else:
        print("CUDA not available. Running on CPU. DataParallel not used.")

    best_loss = float('inf')
    best_model_path = None
    all_losses = []  # 记录所有epoch的loss

    # 训练配置：前90轮只记录loss，后10轮保存模型
    save_start_epoch = 90  # 从第91轮开始保存模型

    print(f"Training for {num_epochs} epochs:")
    print(f"- Epochs 1-{save_start_epoch}: Only recording loss values")
    print(f"- Epochs {save_start_epoch+1}-{num_epochs}: Recording loss + saving models")

    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(progress_bar):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss

            if num_gpus > 1:
                loss = loss.mean()

            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            progress_bar.set_postfix({'loss': loss.item(), 'avg_epoch_loss': total_loss / (step + 1)})

        avg_train_loss = total_loss / len(train_dataloader)
        all_losses.append(avg_train_loss)

        print(f"Epoch: {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}")

        # 前90轮只记录loss，不保存模型
        if epoch < save_start_epoch:
            print(f"  -> Loss recorded (no model saved for epochs 1-{save_start_epoch})")
            continue

        # 后10轮：记录loss并保存模型
        print(f"  -> Saving model for epoch {epoch + 1}")

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        # 保存当前epoch的模型
        current_model_dir = os.path.join(args.save_path, f"model_epoch_{epoch+1}_loss_{avg_train_loss:.4f}")
        if not os.path.exists(current_model_dir):
            os.makedirs(current_model_dir)

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(current_model_dir)
        tokenizer.save_pretrained(current_model_dir)
        print(f"  -> Model saved to: {current_model_dir}")

        # 更新最佳模型
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_model_path = current_model_dir
            print(f"  -> New best model! Loss: {best_loss:.4f}")

    # 保存loss历史记录
    loss_history_file = os.path.join(args.save_path, "loss_history.txt")
    with open(loss_history_file, 'w', encoding='utf-8') as f:
        f.write("Epoch\tLoss\n")
        for i, loss in enumerate(all_losses):
            f.write(f"{i+1}\t{loss:.6f}\n")
    print(f"Loss history saved to: {loss_history_file}")

    print(f"\n=== Training Complete ===")
    print(f"Total epochs: {num_epochs}")
    print(f"Epochs 1-{save_start_epoch}: Loss recorded only")
    print(f"Epochs {save_start_epoch+1}-{num_epochs}: Models saved")
    print(f"Models saved: {num_epochs - save_start_epoch}")

    if best_model_path:
        print(f"Best model: {best_model_path}")
        print(f"Best loss: {best_loss:.4f}")
    else:
        print("No models were saved (all epochs were in recording-only phase)")

    print(f"Model trained on mixed consensus data from full datasets")
    print(f"Using GPT-4o-2024-11-20 generated consensus data")

    # 显示loss趋势
    print(f"\nLoss trend:")
    print(f"First 10 epochs: {[f'{loss:.4f}' for loss in all_losses[:10]]}")
    if len(all_losses) > 10:
        print(f"Last 10 epochs: {[f'{loss:.4f}' for loss in all_losses[-10:]]}")
    print(f"Minimum loss: {min(all_losses):.4f} at epoch {all_losses.index(min(all_losses))+1}")

if __name__ == "__main__":
    main()
