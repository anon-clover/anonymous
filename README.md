# anonymous-submission


## 0. Generate consensus for training dataset
```bash

python openai_generate_consensus_data.py --dataset arc_challenge

```

## 1. Training Consensusor

```bash


CUDA_VISIBLE_DEVICES=3 python training/train_full_consensus.py \
    --train_file data/consensus_training/arc_challenge_consensus_data.jsonl \
    --save_path models/t5_arc_challenge_consensus_100epochs_gpt4o \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 3e-5 \
    --seed 42

```


# ----------------------------------------------------------------------------------------------------------------

## 2. 

### 2.1 
```bash
cd /workspace/
chmod +x scripts/generate_consensus_evidence_variants.sh
./scripts/generate_consensus_evidence_variants.sh
```



### 2.2 

```bash
python data_processing/generate_enhanced_consensus.py \
    --input_file data/popqa/splits/popqa_full_for_application.jsonl \
    --output_file data/enhanced_eval/popqa_full_enhanced_consensus.jsonl \
    --consensus_model_path models/t5_popqa_consensus_100epochs_gpt4o/model_epoch_ \
    --task_type popqa \
    --max_evidence 2 \
    --min_evidence 1 \
    --device cuda:1 \
    --complementarity_threshold 0.95

```


# -----------------------------------------------------------------------------------------------------------------

## 3. inference

#### 
```bash
# PopQA - Enhanced
python inference/selfrag_adaptive_rag.py \
    --input_file data/enhanced_eval/popqa_full.jsonl \
    --output_file results/popqa/popqa.jsonl \
    --selfrag_model_path /root/\
    --task popqa \
    --max_tokens 80 \
    --temperature 0.0 \
    --device cuda:3

```

#### Baseline
```bash
python inference/selfrag_baseline_rag.py \
    --input_file data/enhanced_eval/popqa_full.jsonl \
    --output_file results/popqa/popqa.jsonl \
    --selfrag_model_path /root/ \
    --task popqa \
    --max_tokens 100 \
    --temperature 0.0 \
    --device cuda:2
```

# -----------------------------------------------------------------------------------------------------------------

## 4. eval

### 4.1 PopQA  

```bash
python evaluation/eval.py \
    --golden_file eval_data/popqa_longtail_w_gs.jsonl \
    --answer_file results/popqa/popqa.jsonl \
    --dataset POPQA
```


### 4.4 Bio
```bash

conda activate factscore

python evaluation/evaluate_bio_factscore.py \
    --answer_file results/bio/bio_selfrag_baseline_final_results.jsonl \
    --dataset BIO \
    --openai_api_key YOUR_OPENAI_API_KEY
```

# -----------------------------------------------------------------------------------------------------------------
```

### 5.4 outputs
```
results/
├── popqa/
│   ├── popqa_selfrag_enhanced_final_results.jsonl
│   └── popqa_selfrag_baseline_final_results.jsonl
├── pubqa/
│   ├── pubqa_selfrag_enhanced_final_results.jsonl
│   └── pubqa_selfrag_baseline_final_results.jsonl
├── arc_challenge/
│   ├── arc_challenge_selfrag_enhanced_final_results.jsonl
│   └── arc_challenge_selfrag_baseline_results.jsonl
└── bio/
    ├── bio_selfrag_enhanced_final_results.jsonl
    └── bio_selfrag_baseline_final_results.jsonl
```
