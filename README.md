# anonymous-submission

## 1.1 training
```bash
cd /workspace/conRAG
CUDA_VISIBLE_DEVICES=0 python training/train_full_consensus.py \
    --train_file data/consensus_training/mixed_consensus_for_t5_training_gpt4o_2024_11_20.txt \
    --save_path models/t5_mixed_consensus_100epochs_gpt4o_2024_11_20 \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 3e-5 \
    --seed 42
```


## 2. genneration

### 2.1 
```bash consensus_evidence
cd /workspace/conRAG
chmod +x scripts/generate_consensus_evidence_variants.sh
./scripts/generate_consensus_evidence_variants.sh
```

python data_processing/generate_enhanced_consensus.py \
    --input_file data/popqa/splits/popqa_full_for_application.jsonl \
    --output_file data/enhanced_eval/popqa_full_enhanced_consensus_evidence_top2_quality_kmedoids.jsonl \
    --consensus_model_path models/t5_mixed_consensus_100epochs_gpt4o_2024_11_20/model_epoch_98_loss_0.0069 \
    --task_type popqa \
    --max_evidence 2 \
    --device cuda:0 \
    --complementarity_threshold 0.95
